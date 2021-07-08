import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# title of the app
st.title("Data Visualization App")

# Add a sidebar
st.sidebar.subheader("Visualization Settings")

# Setup file upload
uploaded_file = st.sidebar.file_uploader(
                        label="Upload your CSV or Excel file. (200MB max)",
                         type=['csv', 'xlsx', 'xls'])

global df
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

global numeric_columns
global non_numeric_columns
try:
    st.subheader("First 15 Rows")
    st.write(df.head(15))
    numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
    non_numeric_columns = list(df.select_dtypes(['object']).columns)
    non_numeric_columns.append(None)
    print(non_numeric_columns)
except Exception as e:
    print(e)
    st.write("Please upload file to the application.")

# add a select widget to the side bar
#Data Analysis Selection
DA_select = st.sidebar.selectbox(
    label="Select the Data Analysis you want to see",
    options=['Charts/EDA', 'Machine Learning', 'Sentiment Analysis']
)

#Charts and EDA Choices
if DA_select =='Charts/EDA':
    st.sidebar.subheader("Chart Types")
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot','Heatmap', 'Contour Plot', 'Pie Chart']
    )

    if chart_select == 'Scatterplots':
        st.sidebar.subheader("Scatterplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
            # display the chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Lineplots':
        st.sidebar.subheader("Line Plot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.line(data_frame=df, x=x_values, y=y_values, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        try:
            x = st.sidebar.selectbox('Feature', options=numeric_columns)
            bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                        max_value=100, value=40)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.histogram(x=x, data_frame=df, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        try:
            y = st.sidebar.selectbox("Y axis", options=numeric_columns)
            x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.box(data_frame=df, y=y, x=x, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Heatmap':
        try:
            fig, ax = plt.subplots()
            matrix = np.triu(df.corr())
            plot = sns.heatmap(df.corr(), annot=True,mask=matrix, cmap="rocket")
            st.write(fig)
        except Exception as e:
            print(e)
    if chart_select == 'Contour Plot':
        st.sidebar.subheader("Contour Plot Settings")
        try:
            y = st.sidebar.selectbox("Y axis", options=numeric_columns)
            x = st.sidebar.selectbox("X axis", options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.density_contour(data_frame=df, y=y, x=x, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Pie Chart':
        st.sidebar.subheader("Pie Chart Settings")
        try:
            cols = st.sidebar.selectbox("Attribute", options=non_numeric_columns, format_func=lambda x: x)
            if cols:
                labels = df[cols].value_counts().to_frame().index.tolist()
                values = df[cols].value_counts().tolist()
                plot = px.pie(data_frame=df, values=values, names=labels)
                st.plotly_chart(plot)
        except Exception as e:
            print(e)

#Machine Learning 
if DA_select == 'Machine Learning':
    try:
        st.sidebar.subheader("Machine Learning Models")
        ML_select = st.sidebar.selectbox(
        label="Select the Machine Models",
        options=['Logistic Regression', 'Support Vector Machine', 'Kernel Support Vector Machine',
                    'Training K-Nearest Neighbours', 'Decision Trees', 'Naive Bayes', 'Random Forest']
        )

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

        if (ML_select == "Logistic Regression"):
            predictor('lr', {'penalty': 'l1', 'solver': 'saga', 'max_iter': 5000})

        elif (ML_select == "Support Vector Machine"):
            predictor('svm', {'C': 1, 'gamma': 0.8,'kernel': 'linear', 'random_state': 0})

        elif (ML_select == "Kernel Support Vector Machine"):
            predictor('ksvm', {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'random_state': 0})

        elif (ML_select == "Training K-Nearest Neighbours"):
            predictor('knn', {'n_neighbors': 5, 'n_jobs':1})

        elif (ML_select == "Decision Trees"):
            predictor('dt', {'criterion': 'gini', 'max_features': 'auto', 'splitter': 'random' ,'random_state': 0})

        elif (ML_select == "Naive Bayes"):
            predictor('nb', {})

        elif (ML_select == "Random Forest"):
            predictor('rfc', {'criterion': 'entropy', 'max_features': 'auto', 'n_estimators': 250,'random_state': 0})
    
    except Exception as e:
        print(e)