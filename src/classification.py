import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def classification(df, non_numeric_columns):
    @st.cache(persist=True) #tells the app to cache to disk unless the input or function name changes
    def load_data(data):
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df, keyCol):
        y = df[keyCol]
        x = df.drop(keyCol, axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        # >>> fig, ax = plt.subplots()
        # >>> ax.scatter([1, 2, 3], [1, 2, 3])
        # >>>    ... other plotting actions ...
        # >>> st.pyplot(fig)
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    st.set_option('deprecation.showPyplotGlobalUse', False)

    keyCol = st.selectbox(
        label = "Choose a column to be the key column",
        options = non_numeric_columns,
    )

    class_names = df[keyCol].unique().tolist()
    df = load_data(df)
    x_train, x_test, y_train, y_test = split(df, keyCol)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("SVM", "Logistic Regression", "Random Forest"))
    metrics_list = ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve']

    if classifier == "SVM":
        st.sidebar.subheader('Model hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (kernel coefficient", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", metrics_list)

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: " , accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader('Model hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_lr")
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", metrics_list)

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: " , accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Random Forest":
        st.sidebar.subheader('Model hyperparameters')
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootsrap samples when building trees", ('True', 'False'), key="bootstrap")
        metrics = st.sidebar.multiselect("What metrics to plot?", metrics_list)

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap
            )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: " , accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)