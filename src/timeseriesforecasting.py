# ==================================================================================================
# Import Libraries
# ==================================================================================================
import math
import time
import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import streamlit as st
import plotly_express as px

from tqdm import tqdm
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV, KFold, RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

import datetime
from datetime import datetime, timedelta
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

# ==================================================================================================
# Create Features
# ==================================================================================================
def create_features(df, label=None, persist_cols=None):
    """
    Create time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['amount'] = df[label]

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear', 'amount'] + persist_cols]
    return X

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))
                    ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# ==================================================================================================
# PH Business Calendar
# ==================================================================================================
class PHBusinessCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1),
        Holiday('People Power Anniversary', month=2, day=25),

        Holiday('World Wildlife Day', month=3, day=3),
        Holiday('Day of Valor', month=4, day=9),
        Holiday('Maundy Thursday', month=4, day=9, offset=[Easter(), Day(-3)]),
        Holiday('Good Friday', month=4, day=10, offset=[Easter(), Day(-2)]),
        Holiday('Black Saturday', month=4, day=11, offset=[Easter(), Day(-1)]),

        Holiday('Labor Day', month=5, day=1),
        Holiday('Eid al-Fitr', month=5, day=25),

        Holiday('Philippines Independence Day', month=6, day=12),

        Holiday('Eid al-Adha', month=7, day=31),

        Holiday('Ninoy Aquino Day', month=8, day=21),
        Holiday('National Heroes Day', month=8, day=31),

        Holiday('All Saints Day', month=11, day=1),
        Holiday('All Souls Day', month=11, day=2),
        Holiday('Bonifacio Day', month=11, day=30),

        Holiday('Immaculate Conception', month=12, day=8),
        Holiday('Christmas Eve', month=12, day=24),
        Holiday('Christmas Day', month=12, day=25),
        Holiday('Rizal Day', month=12, day=30),
        Holiday('New Year Eve', month=12, day=31)
    ]

# ==================================================================================================
# Data Cleaning
# ==================================================================================================
def cleaning(dataframe):
    df = dataframe
    df = df[["zhdr_lpro.lprodat", "zhdr_lpro.loanamt", "zhdr_lpro.loantype", "zhdr_lpro.lnintratepa", "zhdr_lpro.lfeffintrate",
             "zhdr_lpro.dgranted", "zhdr_lpro.loanptrate"]]
    
    # Rename columns
    df.columns = ['date_loan', 'amount', 'type', 'addon', 'nominal', 'date_granted', 'term']
    
    # Exclude the rows with 0 amount and 0 date granted
    df = df[df['amount'] != 0]
    df = df[df['date_granted'] != 0]
    df.dropna(inplace = True, subset = ['term'])
    
    # Drop date granted column
    df.drop('date_granted', axis=1, inplace=True)
    
    # Set date_loan as index
    df.set_index('date_loan', drop=True, inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # New columns to be defined
    cols = ['type', 'term', 'addon', 'nominal']
    
    # Create features
    df2 = create_features(df, label='amount', persist_cols=cols)
    df2.reset_index(drop=False, inplace=True)
    df2.sort_values(by='date_loan', inplace=True)
    df2.set_index('date_loan', inplace=True)
    
    # Data Cleaning
    df3 = df2[['type', 'term', 'addon', 'amount']]
    df3['term'] = df3['term'].astype('float')
    df3.reset_index(inplace=True)
    
    # Further pre-processing
    df4 = pd.DataFrame(df3.groupby(['date_loan', 'type', 'term', 'addon'])[
                       'amount'].sum()).reset_index()
    df4['term'] = df4['term'].astype('str')
    df4['addon'] = df4['addon'].astype('str')
    
    # Counts and shares
    df4_counts = df4.set_index('date_loan')[
        'type'].str.get_dummies().sum(level=0)
    df4_dayshares = df4_counts.div(
        df4_counts.sum(axis=1), axis=0).mul(100).round(3)
    
    df4_dayshares_cols = df4_dayshares.columns.tolist()
    df4_dayshares_newcols = [col + '%' for col in df4_dayshares_cols]
    df4_dayshares.columns = df4_dayshares_newcols
    
    # Amounts
    df4_amounts = df4.groupby(['date_loan', 'type'])[
        'amount'].sum().reset_index()
    df4_amounts_new = df4_amounts.pivot(
        index='date_loan', columns='type', values='amount')
    df4_amounts_new = df4_amounts_new.reset_index().rename_axis(
        None, axis=1).set_index('date_loan')
    
    # Terms
    df4_terms = df4.set_index('date_loan')[
        'term'].str.get_dummies().sum(level=0)
    df4_termshares = df4_terms.div(
        df4_terms.sum(axis=1), axis=0).mul(100).round(2)
    
    # Addons
    df4_addons = df4.set_index('date_loan')[
        'addon'].str.get_dummies().sum(level=0)
    df4_addonhares = df4_addons.div(
        df4_addons.sum(axis=1), axis=0).mul(100).round(2)
    
    # Amounts Continuation
    df4_amounts_zeroes = df4_amounts_new.fillna(0)
    df4_amounts_zeroes['total'] = df4_amounts_zeroes.sum(axis=1)
    
    # Concatenation
    df_combined1 = pd.concat(
        [df4_dayshares, df4_termshares, df4_addonhares, df4_amounts_zeroes['total']], axis=1)
    
    df_combined1[df_combined1.columns.tolist()[:-1]] = df_combined1[df_combined1.columns.tolist()[:-1]].shift(35)
    df_combined1['lag_35'] = df_combined1['total'].shift(35)
    df_combined1.dropna(inplace=True)

    df_combined1 = df_combined1.resample('D').sum()
    df_combined1['date'] = df_combined1.index
    df_combined1['dayofweek'] = df_combined1['date'].dt.dayofweek
    df_combined1['quarter'] = df_combined1['date'].dt.quarter
    df_combined1['month'] = df_combined1['date'].dt.month
    df_combined1['dayofyear'] = df_combined1['date'].dt.dayofyear
    df_combined1['dayofmonth'] = df_combined1['date'].dt.day
    df_combined1['weekofyear'] = df_combined1['date'].dt.weekofyear
    
    return df_combined1

# ==================================================================================================
# Data Scaling
# ==================================================================================================
def scale_data(data, test_size=0.05):
    
    X = data.drop(['total'], axis=1)
    y= data['total']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1,shuffle=False)
       
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# ==================================================================================================
# Models
# ==================================================================================================
### XGBOOST
def xgboost(X_train, X_test, y_train, y_test):

    # Model
    xgboost = xgb.XGBRegressor(n_estimators=1500, objective='reg:squarederror', reg_alpha=0.5).fit(X_train,y_train)
    pred = xgboost.predict(X_test)
    
    #Cross validation
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(xgboost, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(xgboost, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(xgboost, X_test, y_test, scoring='r2', cv=cv)
    
    #Scores
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['xgboost'])
    y_pred =pd.DataFrame(pred, columns=['xgboost'])

    return xgboost, y_pred ,df_scores
    
### LINEAR REGRESSION
def Linreg(X_train, X_test, y_train, y_test):
    
    Linreg = LinearRegression().fit(X_train,y_train)
    pred = Linreg.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(Linreg, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(Linreg, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(Linreg, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['Linreg'])
    y_pred = pd.DataFrame(pred, columns=['Linreg'])  

    return Linreg, y_pred ,df_scores

### RANDOM FOREST REGRESSOR
def RFReg(X_train, X_test, y_train, y_test):
    
    RFReg = RandomForestRegressor(n_estimators=1000).fit(X_train,y_train)
    pred = RFReg.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(RFReg, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(RFReg, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(RFReg, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['RandomForestReg'])
    y_pred =pd.DataFrame(pred, columns=['RandomForestReg'])  

    return RFReg, y_pred ,df_scores

### K-NEIGHBORS REGRESSOR
def knn(X_train, X_test, y_train, y_test):
    
    knn = KNeighborsRegressor(n_neighbors=20,leaf_size=100).fit(X_train,y_train)
    pred = knn.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(knn, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(knn, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(knn, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['KNeighborsReg'])
    y_pred =pd.DataFrame(pred, columns=['KNeighborsReg'])  

    return knn, y_pred ,df_scores

### DECISION TREE REGRESSOR
def DTree(X_train, X_test, y_train, y_test):
    
    DTree = DecisionTreeRegressor(random_state=0).fit(X_train,y_train)
    pred = DTree.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(DTree, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(DTree, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(DTree, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['DecisionTreeReg'])
    y_pred =pd.DataFrame(pred, columns=['DecisionTreeReg'])  

    return DTree, y_pred ,df_scores

### SUPPORT VECTOR MACHINE
def SVMreg(X_train, X_test, y_train, y_test):
    
    SVMreg = SVR(kernel='rbf').fit(X_train,y_train)
    pred = SVMreg.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(SVMreg, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(SVMreg, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(SVMreg, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['SupportVectorReg'])
    y_pred =pd.DataFrame(pred, columns=['SupportVectorReg'])  

    return SVMreg, y_pred ,df_scores

### LIGHT GBM
def lgbmR(X_train, X_test, y_train, y_test):
    
    lgbmR = LGBMRegressor().fit(X_train,y_train)
    pred = lgbmR.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(lgbmR, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(lgbmR, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(lgbmR, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['LGBMRegressor'])
    y_pred =pd.DataFrame(pred, columns=['LGBMRegressor'])  

    return lgbmR, y_pred ,df_scores

### MULTI-LAYER PERCEPTRON REG
def MLPreg(X_train, X_test, y_train, y_test):
    
    MLPreg = MLPRegressor(random_state=1, max_iter=1500).fit(X_train,y_train)
    pred = MLPreg.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(MLPreg, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(MLPreg, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(MLPreg, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['MLPRegressor'])
    y_pred =pd.DataFrame(pred, columns=['MLPRegressor'])  

    return MLPreg, y_pred ,df_scores

### VOTING ENSEMBLE
def voting_ensemble(X_train, X_test, y_train, y_test):
    
    # create the sub models
    voting_estimators = []

    # model 1
    Linreg = LinearRegression()
    voting_estimators.append(('LinearRegression', Linreg))
    # model 2
    knn = KNeighborsRegressor(n_neighbors=20,leaf_size=100)
    voting_estimators.append(('KNeighborsRegressor', knn))
    # model 3
    dtree = DecisionTreeRegressor(random_state=0)
    voting_estimators.append(('DecisionTreeRegressor', dtree))
    # model 4
    svmR = SVR(kernel='rbf')
    voting_estimators.append(('SVMRegressor', svmR))
    # model 5
    xgboost = xgb.XGBRegressor(n_estimators=1500, objective='reg:squarederror', reg_alpha=0.5)
    voting_estimators.append(('XGBRegressor', xgboost))
    #model 6
    lgbmR = LGBMRegressor()
    voting_estimators.append(('LGBMRegressor', lgbmR))
    #model 7
    rfreg = RandomForestRegressor(n_estimators=1000)
    voting_estimators.append(('RandomForestRegressor', rfreg))
    #model 8
    MLPreg = MLPRegressor(random_state=1, max_iter=1500)
    voting_estimators.append(('MLPRegressor', MLPreg))
    
    # create the ensemble model
    voting_ensemble = VotingRegressor(voting_estimators).fit(X_train, y_train)
    pred = voting_ensemble.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(voting_ensemble, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(voting_ensemble, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(voting_ensemble, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['voting_ensemble'])
    y_pred =pd.DataFrame(pred, columns=['voting_ensemble'])  

    return voting_ensemble, y_pred ,df_scores

### STACKING ENSEMBLE
def stacking_ensemble(X_train, X_test, y_train, y_test):
    
    # create the sub models
    stacking_estimators_initial = []

    #Base Estimator
    # model 1
    Linreg = LinearRegression()
    stacking_estimators_initial.append(('LinearRegression', Linreg))
    # model 2
    knn = KNeighborsRegressor(n_neighbors=20,leaf_size=100)
    stacking_estimators_initial.append(('KNeighborsRegressor', knn))
    # model 3
    dtree = DecisionTreeRegressor(random_state=0)
    stacking_estimators_initial.append(('DecisionTreeRegressor', dtree))
    # model 4
    svmR = SVR(kernel='rbf')
    stacking_estimators_initial.append(('SVMRegressor', svmR))
    # model 5
    xgboost = xgb.XGBRegressor(n_estimators=1500, objective='reg:squarederror', reg_alpha=0.5)
    stacking_estimators_initial.append(('XGBRegressor', xgboost))
    #model 6
    lgbmR = LGBMRegressor()
    stacking_estimators_initial.append(('LGBMRegressor', lgbmR))
    #model 7
    rfreg = RandomForestRegressor(n_estimators=1000)
    stacking_estimators_initial.append(('RandomForestRegressor', rfreg))
    #model 8
    MLPreg = MLPRegressor(random_state=1, max_iter=1500)
    stacking_estimators_initial.append(('MLPRegressor', MLPreg))

    #Final/Generalize Estimator
    stacking_estimators_final = LinearRegression()
    
    stacking_ensemble = StackingRegressor(estimators=stacking_estimators_initial, 
                          final_estimator=stacking_estimators_final, cv=5).fit(X_train,y_train)
    pred = stacking_ensemble.predict(X_test)
    
    cv = KFold(n_splits=10, random_state=7,shuffle=True)
    MAE = cross_val_score(stacking_ensemble, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
    RSME = cross_val_score(stacking_ensemble, X_test, y_test, scoring='neg_root_mean_squared_error', cv=cv)
    R2 = cross_val_score(stacking_ensemble, X_test, y_test, scoring='r2', cv=cv)
    
    scores = {'MEA':np.mean(MAE), 'RSME':np.mean(RSME), 'R2':np.mean(R2)}
    df_scores = pd.DataFrame(scores, index=['stacking_ensemble'])
    y_pred =pd.DataFrame(pred, columns=['stacking_ensemble'])  

    return stacking_ensemble, y_pred ,df_scores

# ==================================================================================================
# All Models
# ==================================================================================================
def all_models(X_train, X_test, y_train, y_test):
    xgboost_model, xgboost_pred, xgboost_scores    =  xgboost(X_train, X_test, y_train, y_test)
    Linreg_model, Linreg_pred, Linreg_scores       =  Linreg(X_train, X_test, y_train, y_test)
    RFReg_model, RFReg_pred, RFReg_scores          =  RFReg(X_train, X_test, y_train, y_test)
    knn_model, knn_pred, knn_scores                =  knn(X_train, X_test, y_train, y_test)
    DTree_model, DTree_pred, DTree_scores          =  DTree(X_train, X_test, y_train, y_test)
    SVMreg_model, SVMreg_pred, SVMreg_scores       =  SVMreg(X_train, X_test, y_train, y_test)
    lgbmR_model, lgbmR_pred, lgbmR_scores          =  lgbmR(X_train, X_test, y_train, y_test)
    MLPreg_model, MLPreg_pred, MLPreg_scores       =  MLPreg(X_train, X_test, y_train, y_test)
    voting_model, voting_pred, voting_scores       =  voting_ensemble(X_train, X_test, y_train, y_test)
    stacking_model, stacking_pred, stacking_scores =  stacking_ensemble(X_train, X_test, y_train, y_test)
    
    #Predictions
    y_pred_list = [xgboost_pred, Linreg_pred, RFReg_pred, knn_pred, DTree_pred, SVMreg_pred, lgbmR_pred, MLPreg_pred,
                   voting_pred, stacking_pred]

    y_pred = pd.concat(y_pred_list, axis=1)
    
    #y_test
    predictions= pd.DataFrame(y_test)
    predictions = predictions.reset_index()
    predictions.columns = ['date_loan','Test']
    
    #Scores
    scores_list = [xgboost_scores, Linreg_scores, RFReg_scores, knn_scores, DTree_scores, SVMreg_scores, lgbmR_scores, MLPreg_scores,
                   voting_scores, stacking_scores]

    df_scores = pd.concat(scores_list, axis=0).sort_values(by=['R2'], ascending=False)
    best_scores = pd.DataFrame(df_scores.iloc[0])
    col = best_scores.columns
    
    #Plot
    predictions['Predicted'] = y_pred[col]
    test_data = predictions['Test'].tail(30)
    pred_data = predictions['Predicted'].tail(30)
    
    res = test_data - pred_data
    res_std = res.std() 
    y_upper = pred_data + (1.96 * res_std)
    y_lower = pred_data - (1.96 * res_std)
    # 1.96 is an arbitrary number based on 95% CI
    # Source: https://otexts.com/fpp2/prediction-intervals.html
    
    return df_scores, best_scores, col, pred_data, test_data, y_lower, y_upper
  
# ==================================================================================================
# Forecast
# ==================================================================================================
#Create the bar graph that compares the accuracy of each models
def createFig(values):
    modelDf = pd.Series(values, name='Accuracy Score')
    modelDf.index.name = 'Model'
    modelDf.reset_index()
    fig1 = px.bar(modelDf, x=modelDf.index, y=modelDf.name)
    st.write(fig1)

def forecast(data):
    df = cleaning(data)

    df2 = df.resample('D').sum()
    X_train, X_test, y_train, y_test = scale_data(df2, test_size=0.05)

    st.header('Predictions')
    df_scores, best_scores, best_model, pred_data, test_data, y_lower, y_upper = all_models(X_train, X_test, y_train, y_test)

    st.header('Model Performance Comparison')
    st.write(df_scores)
    createFig(df_scores[df_scores['R2'] > 0]['R2'])

    st.header('Best Model Performance')
    #st.subheader(best_model.iloc[0:0])
    st.write(best_scores)

    fig, ax = plt.subplots()
    ax.plot(pred_data, color='blue', marker='o', label='Predicted')
    ax.plot(test_data, color='orange', marker='o', label='Test')
    ax.set_title('Time Series Forecast - Predicted vs Test')
    #plt.rcParams["xtick.labelsize"] = 5
    plt.rcParams["figure.figsize"] = (8, 4)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(pred_data, color='blue', marker='o', label='Predicted')
    ax.fill_between(pred_data.index, y_lower, y_upper, alpha=0.3)
    ax.set_title('Time Series Forecast with Prediction Intervals')
    #plt.rcParams["xtick.labelsize"] = 5
    plt.rcParams["figure.figsize"] = (8, 4)
    st.pyplot(fig)


### Streamlit version of line graphs (1 chart for test, 1 chart for pred)
    #st.line_chart(test_data)
    #st.line_chart(pred_data)


### For selecting timeframe of predictions
#    st.sidebar.subheader("Select Timeframe")
#    timeframe_select = st.sidebar.selectbox(
#        label="Select Timeframe",
#        options=['3 months', '6 months', '1 year', '2 years']
#    )

#    if timeframe_select == '3 months':
#        time = 90
#        tomorrow = data.index[-1:][0] + timedelta(days=1)
#        future = data.index[-1:][0] + timedelta(days=time)
    
#    if timeframe_select == '6 months':
#        time = 180
#        tomorrow = data.index[-1:][0] + timedelta(days=1)
#        future = data.index[-1:][0] + timedelta(days=time)
    
#    if timeframe_select == '1 year':
#        time = 365
#        tomorrow = data.index[-1:][0] + timedelta(days=1)
#        future = data.index[-1:][0] + timedelta(days=time)
    
#    if timeframe_select == '2 years':
#        time = 730
#        tomorrow = data.index[-1:][0] + timedelta(days=1)
#        future = data.index[-1:][0] + timedelta(days=time)


### For selecting confidence level
#    st.sidebar.subheader("Select Confidence Level")
#    confidence_level_select = st.sidebar.selectbox(
#        label="Select Confidence Level",
#        options=['90%', '95%', '99%']
#    )


### For computing prediction intervals using dynamic multiplier
#    alpha = 0.05
#    bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
#    q_bootstrap = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)

#    if confidence_level_select == "90%":
#        confidence_level = 1.645

#    if confidence_level_select == "95%":
#        confidence_level = 1.96

#    if confidence_level_select == "99%":
#        confidence_level = 2.576


### For outputting dfs of prediction, upper and lower values
#    st.header("Prediction Values")
#    st.dataframe(predictions)
#    st.header("Predicted Upper Values")
#    st.write(y_upper)
#    st.header("Predicted Lower Values")
#    st.write(y_lower)
