import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

model_scores = {}
accuracy_scores = {}
def show_machinelearning_analysis(df):
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
                #st.title('Training Logistic Regression on Training Set')
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**params)

            elif predictor == 'SVC':
                #st.title('Training Support Vector Machine on Training Set')
                from sklearn.svm import SVC
                model = SVC(**params)
            
            elif predictor == 'K-SVC':
                #st.title('Training Kernel Support Vector Machine on Training Set')
                from sklearn.svm import SVC
                model = SVC(**params)

            elif predictor == 'KNN':
                #st.title('Training K-Nearest Neighbours on Training Set')
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(**params)

            elif predictor == 'Decision Tree':
                #st.title('Training Decision Tree Model on Training Set')
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(**params)

            elif predictor == 'Gaussian NB':
                #st.title('Training Naive Bayes Model on Training Set')
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB(**params)
                
            elif predictor == 'Random Forest':
                #st.title('Training Random Forest Model on Training Set')
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)
            else:
                #st.write('Invalid Predictor!')
                exit    

            model.fit(X_train, y_train)
            
            #filename = "something_" + predictor + ".pkl"
            #pickle.dump(model, open(filename, 'wb'))

            #st.subheader('''Prediciting Test Set Result''')
            y_pred = model.predict(X_test)
            result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
            #st.write(result,'\n')

            #st.subheader('''Making Confusion Matrix''')
            from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            #st.write(cm,'\n')

            #st.subheader('''Classification Report''')
            #st.text('Model Report:\n ' + classification_report(y_test, y_pred))

           # st.subheader('''Evaluating Model Performance''')
            accuracy = accuracy_score(y_test, y_pred)
            #st.write("Accuracy: {:.2f} %".format(accuracy.mean()*100))
            model_scores[predictor] = accuracy.mean()*100
            #st.subheader('''Applying K-Fold Cross validation''')
            from sklearn.model_selection import cross_val_score
            accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
            #st.write("Accuracy: {:.2f} %".format(accuracies.mean()*100))
            accuracy_scores[predictor] = accuracies.mean()*100
            #st.write("Standard Deviation: {:.2f} %".format(accuracies.std()*100),'\n') 
        
        predictor('LogisticRegression', {'penalty': 'l1', 'solver': 'saga', 'max_iter': 5000})
        predictor('SVC', {'C': 1, 'gamma': 0.8,'kernel': 'linear', 'random_state': 0})
        predictor('K-SVC', {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'random_state': 0})
        predictor('KNN', {'n_neighbors': 5, 'n_jobs':1})
        predictor('Decision Tree', {'criterion': 'gini', 'max_features': 'auto', 'splitter': 'random' ,'random_state': 0})
        predictor('Gaussian NB', {})
        predictor('Random Forest', {'criterion': 'entropy', 'max_features': 'auto', 'n_estimators': 250,'random_state': 0})


        def createFig(values):
            modelDf = pd.Series(values, name='Accuracy Score')
            modelDf.index.name = 'Models'
            modelDf.reset_index()
            fig1 = px.bar(modelDf, x=modelDf.index, y=modelDf.name)
            st.write(fig1)
        
        
        st.subheader('Model Performance Comparison')
        createFig(model_scores)
        st.subheader('K-Fold Cross Validation Comparison')
        createFig(accuracy_scores)

    except Exception as e:
        print(e)