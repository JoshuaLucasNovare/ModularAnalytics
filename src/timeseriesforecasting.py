import math
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import os
import streamlit as st


from tqdm import tqdm
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

import datetime
from datetime import datetime, timedelta
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()
#%matplotlib inline


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

def training(data):

    X = data.drop(['total', 'date'], axis=1)
    y = data['total']
    X_train = X.iloc[:len(data)-35]
    X_test = X.iloc[len(data)-35:]
    y_train = y.iloc[:len(data)-35]
    y_test = y.iloc[len(data)-35:]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.85)
    new_Xtrain = pca.fit_transform(X_train)
    new_Xtest = pca.transform(X_test)

    reg = xgb.XGBRegressor(
        n_estimators=1500, objective='reg:squarederror', reg_alpha=0.5)
    reg.fit(new_Xtrain, y_train, verbose=False)

    
    return reg, new_Xtest, y_test

def cleaning(df):

    # Read the data
    chunks = df[["LPRODAT", "LOANTYPE", "DGRANTED", "LNINTRATEPA", "LOANAMT", "LNEFFINTRATE", "LOANPTRATE"]]

    # # Read the data
    # chunks = df(
    #     header=0,
    #     low_memory=False,
    #     delimiter=',',
    #     usecols=["LPRODAT", "LOANAMT", "LOANTYPE", "LNINTRATEPA", "LNEFFINTRATE",
    #             "DGRANTED", "LOANPTRATE"],
    #     chunksize=50000
    # )

    # Append all chunks
    #df = pd.concat([chunk for chunk in chunks])
    df = chunks

    # Convert LPRODAT dtypes to datetime
    df['LPRODAT'] = df['LPRODAT'].apply(
        lambda x: pd.to_datetime(str(x), format='%d/%m/%Y'))
    # Rename columns
    df.columns = ['date_loan', 'type', 'date_granted',
                'term', 'amount', 'addon', 'nominal']

    # Exlclude the rows with 0 amount and 0 date granted
    df = df[df['amount'] != 0]
    df = df[df['date_granted'] != 0]

    # Replace LPT048 with corresponding numeric term
    df['term'] = df['term'].replace('LPT048', 48.0)
    df['term'] = df['term'].replace('LPT030', 30.0)
    df['term'] = df['term'].replace('LPT036', 36.0)
    df['term'] = df['term'].replace('LPT060', 60.0)
    df['term'] = df['term'].replace('LPT024', 24.0)
    df['term'] = df['term'].replace('LPT012', 12.0)
    df['term'] = df['term'].replace('LPT054', 54.0)
    df['term'] = df['term'].replace('LPT042', 42.0)
    df['term'] = df['term'].replace('LPT018', 18.0)


    # Exclude whitespaces term
    df = df[df['term'] != ' ']

    # Drop date granted column
    df.drop('date_granted', axis=1, inplace=True)

    # Set date_loan as index
    df.set_index('date_loan', drop=True, inplace=True)

    # New columns to be defined
    cols = ['type', 'term', 'addon', 'nominal']

    # Create features
    df2 = create_features(df, label='amount', persist_cols=cols)
    df2.reset_index(drop=False, inplace=True)
    df2.sort_values(by='date_loan', inplace=True)
    df2.set_index('date_loan', inplace=True)

    # Get only the 2010 onwards data
    df2 = df2[(df2['year'] >= 2003)]

    # Clean the data
    df3 = df2[['type', 'term', 'addon', 'amount']]
    df3['term'] = df3['term'].astype('float')
    df3.reset_index(inplace=True)

    # Further pre-processing
    df4 = pd.DataFrame(df3.groupby(['date_loan', 'type', 'term', 'addon'])[
                    'amount'].sum()).reset_index()
    df4['term'] = df4['term'].astype('str')
    df4['addon'] = df4['addon'].astype('str')
    df4_counts = df4.set_index('date_loan')[
        'type'].str.get_dummies().sum(level=0)
    df4_dayshares = df4_counts.div(
        df4_counts.sum(axis=1), axis=0).mul(100).round(3)
    df4_dayshares.columns = ['AL%', 'AP%', 'AS%', 'BL%']
    df4_amounts = df4.groupby(['date_loan', 'type'])[
        'amount'].sum().reset_index()
    df4_amounts_new = df4_amounts.pivot(
        index='date_loan', columns='type', values='amount')
    df4_amounts_new = df4_amounts_new.reset_index().rename_axis(
        None, axis=1).set_index('date_loan')
    df4_terms = df4.set_index('date_loan')[
        'term'].str.get_dummies().sum(level=0)
    df4_termshares = df4_terms.div(
        df4_terms.sum(axis=1), axis=0).mul(100).round(2)
    df4 = df4[(df4['addon'] != '156800.0')]
    df4 = df4[(df4['addon'] != '35200.0')]
    df4_addons = df4.set_index('date_loan')[
        'addon'].str.get_dummies().sum(level=0)
    df4_addonhares = df4_addons.div(
        df4_addons.sum(axis=1), axis=0).mul(100).round(2)
    df4_amounts_zeroes = df4_amounts_new.fillna(0)
    df4_amounts_zeroes['total'] = df4_amounts_zeroes.sum(axis=1)
    df_combined1 = pd.concat(
        [df4_dayshares, df4_termshares, df4_addonhares, df4_amounts_zeroes['total']], axis=1)

    return df_combined1

def training(data):

    X = data.drop(['total', 'date'], axis=1)
    y = data['total']
    X_train = X.iloc[:len(data)-time]
    X_test = X.iloc[len(data)-time:]
    y_train = y.iloc[:len(data)-time]
    y_test = y.iloc[len(data)-time:]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.85)
    new_Xtrain = pca.fit_transform(X_train)
    new_Xtest = pca.transform(X_test)

    reg = xgb.XGBRegressor(
        n_estimators=1500, objective='reg:squarederror', reg_alpha=0.5)
    reg.fit(new_Xtrain, y_train, verbose=False)

    
    return reg, new_Xtest, y_test

# def forecast(file_path, f, save_path):
def forecast(df):

    st.sidebar.subheader("Timeframe")
    timeframe_select = st.sidebar.selectbox(
        label="Select timeframe",
        options=['3 months', '6 months', '1 year', '2 years']
    )
    
    global time

    data = cleaning(df)

    latest = data.index[-1:][0]

    if timeframe_select == '3 months':
        time = 90
        tomorrow = data.index[-1:][0] + timedelta(days=1)
        future = data.index[-1:][0] + timedelta(days=time)
    
    if timeframe_select == '6 months':
        time = 180
        tomorrow = data.index[-1:][0] + timedelta(days=1)
        future = data.index[-1:][0] + timedelta(days=time)
    
    if timeframe_select == '1 year':
        time = 365
        tomorrow = data.index[-1:][0] + timedelta(days=1)
        future = data.index[-1:][0] + timedelta(days=time)
    
    if timeframe_select == '2 years':
        time = 730
        tomorrow = data.index[-1:][0] + timedelta(days=1)
        future = data.index[-1:][0] + timedelta(days=time)

    PH_BD = CustomBusinessDay(calendar=PHBusinessCalendar())
    s = pd.date_range(tomorrow, end=future, freq=PH_BD)
    dfs = pd.DataFrame(s, columns=['Date'])
    futuredates = dfs['Date'].tolist()
    cols = data.columns.tolist()

    df_future = pd.DataFrame(index=futuredates, columns=cols)
    df_future.fillna(0, inplace=True)
    df_future.index = pd.to_datetime(df_future.index)

    df_future2 = pd.concat([data, df_future])
    df_future2[df_future2.columns.tolist(
    )[:-1]] = df_future2[df_future2.columns.tolist()[:-1]].shift(35)
    df_future2['lag_35'] = df_future2['total'].shift(35)
    df_future2.dropna(inplace=True)

    df_future3 = df_future2.resample('D').sum()
    df_future3['date'] = df_future3.index
    df_future3['dayofweek'] = df_future3['date'].dt.dayofweek
    df_future3['quarter'] = df_future3['date'].dt.quarter
    df_future3['month'] = df_future3['date'].dt.month
    df_future3['dayofyear'] = df_future3['date'].dt.dayofyear
    df_future3['dayofmonth'] = df_future3['date'].dt.day
    df_future3['weekofyear'] = df_future3['date'].dt.weekofyear

    model, X_test, y_test = training(df_future3)
    y_pred = model.predict(X_test)
    predictions = pd.DataFrame(y_test)
    predictions.columns = ['Test']
    predictions['Predicted'] = y_pred
    predictions = predictions[predictions['Predicted'] > 0]

    res = predictions['Test'] - predictions['Predicted'] # added a residual computation

    predictions.drop('Test', axis=1, inplace=True)

    alpha = 0.05

    bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
    q_bootstrap = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)

    #y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    y_lower = predictions['Predicted'] + q_bootstrap[0].mean()
    y_upper = predictions['Predicted'] + q_bootstrap[1].mean()

    # predictions['Predicted'].plot(legend=True, marker='o')

    #predictions.to_csv('output/prediction.csv', index = False)
#     predictions.to_csv(os.path.join(save_path, 'prediction.csv'))

    #return display
    fig, ax = plt.subplots()
    ax.plot(predictions, marker='o')
    ax.fill_between(predictions.index, y_lower, y_upper, alpha=0.3)
    # st.write(predictions.index)
    # ax.fill_between(predictions.index, predictions['Predicted']-100000, predictions['Predicted']+100000, alpha=0.3)
    ax.set_title('Time Series Forecast') 
    plt.rcParams["xtick.labelsize"] = 5
    # plt.rcParams["figure.figsize"] = (8, 4)
    st.pyplot(fig)
    st.dataframe(predictions)

