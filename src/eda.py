from altair.vegalite.v4.schema.core import BrushConfig
import streamlit as st
import plotly_express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 


def show_eda(df, numeric_columns, non_numeric_columns):
    st.sidebar.subheader("Chart Types")
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot','Heatmap', 'Contour Plot', 'Pie Chart', 'Distplot', 'Trendlines', 'Violin plot', 'Bubble Chart', 'Classification']
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
            st.write('A scatter plot (aka scatter chart, scatter graph) uses dots to represent values for two different numeric variables. Scatter plots are used to observe relationships between variables.')
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
            st.write('A Lineplot (aka line graph) uses connected lines to show data. Lineplots are used to see changes over a period of time.')
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
            st.write('A histogram is a graphical representation that organizes a group of data points into user-specified ranges. Similar in appearance to a bar graph, the histogram condenses a data series into an easily interpreted visual by taking many data points and grouping them into logical ranges or bins.')
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
            st.write('A boxplot is a graph that gives you a good indication of how the values in the data are spread out.')
        except Exception as e:
            print(e)
    if chart_select == 'Heatmap':
        try:
            fig, ax = plt.subplots()
            matrix = np.triu(df.corr())
            plot = sns.heatmap(df.corr(), annot=True,mask=matrix, cmap="rocket")
            st.write(fig)
            st.write('Heat Maps are graphical representations of data that utilize color-coded systems. The primary purpose of Heat Maps is to better visualize the volume of locations/events within a dataset and assist in directing viewers towards areas on data visualizations that matter most.')
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
            st.write('A contour plot is a graphical technique for representing a 3-dimensional surface by plotting constant z slices, called contours, on a 2-dimensional format.')
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
                st.write('A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion.')
        except Exception as e:
            print(e)

    if chart_select == 'Distplot':
        st.subheader("Distplot Settings")
        try:
            feat = st.multiselect(
                label = "Choose features",
                options = range(len(numeric_columns)),
                format_func = lambda x: numeric_columns[x]
            )
            fig = ff.create_distplot([df[numeric_columns[c]] for c in feat], [numeric_columns[x] for x in feat]) # change df['feature'] to df.iloc[~~] since duplicate columns might be possible
            st.plotly_chart(fig, use_container_width=True)
            st.write('A Distplot or distribution plot, depicts the variation in the data distribution. The Distplot depicts the data by a histogram and a line in combination to it.')
        except Exception as e:
            print(e)

    if chart_select == 'Trendlines':
        st.sidebar.subheader("Trendlines Settings")
        type = st.sidebar.selectbox("Linear or Non-Linear?", options = ['Linear', 'Non-Linear'])
        x = st.sidebar.selectbox("Horizontal Axis", options=numeric_columns)
        y = st.sidebar.selectbox('Vertical Axis', options = numeric_columns)
        if type == "Linear":
            try:
                fig = px.scatter(df, x=x, y=y, trendline="ols")
                st.plotly_chart(fig)
            except Exception as e:
                print(e)
        elif type == "Non-Linear":
            try:
                fig = px.scatter(df, x=x, y=y, trendline="lowess")
                st.plotly_chart(fig)
            except Exception as e:
                print(e)
        st.write('A trend line, often referred to as a line of best fit, is a line that is used to represent the behavior of a set of data to determine if there is a certain pattern. Determining if a set of points exhibits a positive trend, a negative trend, or no trend at all.')

    if chart_select == "Violin plot":
        st.sidebar.subheader("Violin plot Settings")
        try:
            feat = st.sidebar.selectbox(
                label = "Choose feature",
                options = range(len(numeric_columns)),
                format_func = lambda x: numeric_columns[x]
            )
            fig = px.violin(df, y=numeric_columns[feat], box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
            )
            st.plotly_chart(fig)
            st.write('A violin plot is a method of plotting numeric data. It is similar to a box plot, with the addition of a rotated kernel density plot on each side. Violin plots are similar to box plots, except that they also show the probability density of the data at different values, usually smoothed by a kernel density estimator.')
        except Exception as e:
            print(e)
    
    if chart_select == "Bubble Chart":
        st.sidebar.subheader("Bubble Chart Settings")
        x = st.sidebar.selectbox(
            label = "X Axis",
            options = range(len(numeric_columns)),
            format_func = lambda a: numeric_columns[a]
        )
        y = st.sidebar.selectbox(
            label = "Y Axis",
            options = range(len(numeric_columns)),
            format_func = lambda b: numeric_columns[b]
        )
        group = st.sidebar.selectbox(
            label = "Group By",
            options = range(len(non_numeric_columns)),
            format_func = lambda c: non_numeric_columns[c]
        )
        size = st.sidebar.selectbox(
            label = "Set Size",
            options = range(len(numeric_columns)),
            format_func = lambda d: numeric_columns[d]
        )
        try:
            fig = px.scatter(df, x=numeric_columns[x], y=numeric_columns[y], 
                color=non_numeric_columns[group], size = numeric_columns[size],
                log_x=True, size_max=60
            )
            st.plotly_chart(fig)
            st.write("A bubble chart is a type of chart that displays three dimensions of data. Each entity with its triplet of associated data is plotted as a disk that expresses two of the vᵢ values through the disk's xy location and the third through its size.")
        except Exception as e:
            print(e)

    if chart_select == "Classification":
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















