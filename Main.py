import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# Web App Title
st.markdown('''
# **The EDA App**
This is the **EDA App** created in Streamlit using the **pandas-profiling** library.
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, minimal=True)
    st.header('**Input DataFrame (First 15 Rows)**')
    st.write(df.head(15))
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)

    X = df.iloc[:,1:].values
    y = df.iloc[:, 0:1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    accuracy_scores = {}
    #Testing Multiple Training Model
    def predictor(predictor, params):
        '''This function is made to test multiple training models'''
        global accuracy_scores
        if predictor == 'lr':
            st.title('Training Logistic Regression on Training Set')
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**params)

        elif predictor == 'svm':
            st.title('Training Support Vector Machine on Training Set')
            from sklearn.svm import SVC
            model = SVC(**params)
        
        elif predictor == 'ksvm':
            st.title('Training Kernel Support Vector Machine on Training Set')
            from sklearn.svm import SVC
            model = SVC(**params)

        elif predictor == 'knn':
            st.title('Training K-Nearest Neighbours on Training Set')
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(**params)

        elif predictor == 'dt':
            st.title('Training Decision Tree Model on Training Set')
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(**params)

        elif predictor == 'nb':
            st.title('Training Naive Bayes Model on Training Set')
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB(**params)
            
        elif predictor == 'rfc':
            st.title('Training Random Forest Model on Training Set')
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
        else:
            st.write('Invalid Predictor!')
            exit    

        model.fit(X_train, y_train)
        #filename = "something_" + predictor + ".pkl"
        #pickle.dump(model, open(filename, 'wb'))

        st.subheader('''Prediciting Test Set Result''')
        y_pred = model.predict(X_test)
        result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
        st.write(result,'\n')

        st.subheader('''Making Confusion Matrix''')
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm,'\n')

        st.subheader('''Classification Report''')
        st.text('Model Report:\n ' + classification_report(y_test, y_pred))

        st.subheader('''Evaluating Model Performance''')
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy: {:.2f} %".format(accuracy.mean()*100))

        st.subheader('''Applying K-Fold Cross validation''')
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
        st.write("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        accuracy_scores[model] = accuracies.mean()*100
        st.write("Standard Deviation: {:.2f} %".format(accuracies.std()*100),'\n') 

    st.subheader('List of Models Available:')
    option = st.radio('What model whould you like to use?',
    ('Logistic Regression', 'Support Vector Machine', 'Kernel Support Vector Machine',
    'Training K-Nearest Neighbours', 'Decision Trees', 'Naive Bayes', 'Random Forest'))

    if (option == "Logistic Regression"):
        predictor('lr', {'penalty': 'l1', 'solver': 'saga', 'max_iter': 5000})

    elif (option == "Support Vector Machine"):
        predictor('svm', {'C': 1, 'gamma': 0.8,'kernel': 'linear', 'random_state': 0})

    elif (option == "Kernel Support Vector Machine"):
        predictor('ksvm', {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'random_state': 0})

    elif (option == "Training K-Nearest Neighbours"):
        predictor('knn', {'n_neighbors': 5, 'n_jobs':1})

    elif (option == "Decision Trees"):
        predictor('dt', {'criterion': 'gini', 'max_features': 'auto', 'splitter': 'random' ,'random_state': 0})

    elif (option == "Naive Bayes"):
        predictor('nb', {})

    elif (option == "Random Forest"):
        predictor('rfc', {'criterion': 'entropy', 'max_features': 'auto', 'n_estimators': 250,'random_state': 0})

else:
    st.info('Awaiting for CSV file to be uploaded.')
