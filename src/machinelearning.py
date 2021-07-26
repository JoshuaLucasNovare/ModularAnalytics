import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

model_scores = {}
accuracy_scores = {}
def show_machinelearning_analysis(df):
    df = df.dropna(axis = 1, how='all')
    columns = list(df.columns)
    keyCol = st.selectbox(
        label = "Choose a column to be the key column",
        options = range(len(columns)),
        format_func = lambda x: columns[x]
    )
    if keyCol != 0:
        columns[keyCol], columns[0] = columns[0], columns[keyCol]
        df = df.reindex(columns=columns)

    not_correlated_features = set()
    testdf = df.corr(method ='kendall') #methods = pearson : standard correlation coefficient; kendall : Kendall Tau correlation coefficient; spearman : Spearman rank correlation

    for i in range(testdf.shape[0]):
        for j in range(i):
            if abs(testdf.iloc[i, j]) > 0.5:
                colname = testdf.columns[i]
                not_correlated_features.add(colname)
    
    st.write(df.drop(columns = not_correlated_features))
    try:   
        X = df.iloc[:,1:].values
        y = df.iloc[:, 0:1].values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        #Testing Multiple Training Model
        def predictor(predictor, params):
            '''This function is made to test multiple training models'''
            global accuracy_scores
            if predictor == 'LogisticRegression':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**params)

            elif predictor == 'SVC':
                from sklearn.svm import SVC
                model = SVC(**params)
            
            elif predictor == 'K-SVC':
                from sklearn.svm import SVC
                model = SVC(**params)

            elif predictor == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(**params)

            elif predictor == 'Decision Tree':
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(**params)

            elif predictor == 'Gaussian NB':
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB(**params)
                
            elif predictor == 'Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)   

            model.fit(X_train, y_train)

            from sklearn.metrics import accuracy_score
            y_pred = model.predict(X_test)

           #Model Performance Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[predictor] = accuracy.mean()*100

            #K-Fold Cross Validation Accuracy
            from sklearn.model_selection import cross_val_score
            accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
            accuracy_scores[predictor] = accuracies.mean()*100
        
        predictor('LogisticRegression', {'penalty': 'l1', 'solver': 'saga', 'max_iter': 5000})
        predictor('SVC', {'C': 1, 'gamma': 0.8,'kernel': 'linear', 'random_state': 0})
        predictor('K-SVC', {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'random_state': 0})
        predictor('KNN', {'n_neighbors': 5, 'n_jobs':1})
        predictor('Decision Tree', {'criterion': 'gini', 'max_features': 'auto', 'splitter': 'random' ,'random_state': 0})
        predictor('Gaussian NB', {})
        predictor('Random Forest', {'criterion': 'entropy', 'max_features': 'auto', 'n_estimators': 250,'random_state': 0})

        #Create the bar graph that compares the accuracy of each models
        def createFig(values):
            modelDf = pd.Series(values, name='Accuracy Score')
            modelDf.index.name = 'Model'
            modelDf.reset_index()
            fig1 = px.bar(modelDf, x=modelDf.index, y=modelDf.name)
            st.write(fig1)
        
        st.subheader('Model Performance Comparison')
        createFig(model_scores)
        st.subheader('K-Fold Cross Validation Comparison')
        createFig(accuracy_scores)

    except Exception as e:
        print(e)