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
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error

import altair as alt
from fbprophet import Prophet
import datetime


def show_eda(df, numeric_columns, non_numeric_columns):
    st.sidebar.subheader("Chart Types")
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot','Heatmap', 'Contour Plot', 'Pie Chart', 'Distplot', 'Trendlines', 'Violin plot', 'Bubble Chart', 'Classification', "Cash Flow Projection"]
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

    if chart_select == "Cash Flow Projection":
        st.sidebar.subheader("Cash Flow Settings")
        cash = st.sidebar.selectbox(
            label = "Select Cash Column",
            options = range(len(df.columns)),
            format_func = lambda x: df.columns[x]
        )
        date = st.sidebar.selectbox(
            label = "Select Date Column",
            options = range(len(df.columns)),
            format_func = lambda y: df.columns[y]
        )
        inflow_outflow = st.sidebar.selectbox(
            label = "Select inflow/outflow column",
            options = range(len(df.columns)),
            format_func = lambda z: df.columns[z]
        )
        try:
            df.iloc[:,cash] = pd.to_numeric(df.iloc[:,cash])
        except Exception as e:
            print(e)
        try:
            df.iloc[:,date] = pd.to_datetime(df.iloc[:,date])
        except Exception as e:
            print(e)

        df['Year'] = df.iloc[:,date].dt.year
        df['Month'] = df.iloc[:,date].dt.month

        st.write(alt.Chart(df).mark_bar().encode(
            alt.X("Year:N"),
            y = 'count()',
        ).properties(height=500, width=600))

        st.write(alt.Chart(df).mark_bar().encode(
            alt.X("Month:N"),
            y = 'count()',
        ).properties(height=500, width=600))

        try:
            cashflow = df.pivot_table(
                index=[df.columns[date], df.columns[inflow_outflow]], 
                values=df.columns[cash], 
                aggfunc='sum'
            ).reset_index()
        except Exception as e:
            print(e)

        df2 = cashflow.pivot_table(
            df.columns[cash], # value
            [df.columns[date]], # index
            df.columns[inflow_outflow] # columns
        ).reset_index()
        df2['cashflow'] = df2['cash_inflow'] - df2['cash_outflow']

        st.write(alt.Chart(df2).mark_bar().encode(
            alt.X("cashflow:Q", bin=alt.Bin(maxbins=70)),
            y = 'count()',
        ).properties(height=500, width=600))

        cash_over_time = df2[df2['cashflow'] != 0]
        #cash_over_time = cash_over_time[cash_over_time["cashflow.transaction_date"] < '2019-01-01']

        cash_over_time.rename(columns={f'{df.columns[date]}': 'date'}, inplace=True)

        st.write(alt.Chart(cash_over_time).mark_line(point=True).encode(
            x='date:T',
            y=alt.Y('cashflow:Q'),
            tooltip=['cashflow', 'date']
        ).properties(height=500, width=1000))

        historical = cash_over_time

        # Prophet Training and Forecasting

        def train_model(data, holidays=False, ws=False, ys=False, ds=False, ms=False, iw=0.85):
            ts = data
            ts.columns = ['ds', 'y']
            
            if isinstance(holidays, pd.DataFrame):
                model = Prophet(weekly_seasonality=ws, yearly_seasonality=ys, daily_seasonality=ds, interval_width=iw, holidays=holidays)
            else:
                model = Prophet(weekly_seasonality=ws, yearly_seasonality=ys, daily_seasonality=ds, interval_width=iw)
            if ms:
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model = model.fit(ts)
            return model


        def predict(model, n):
            forecast = model.make_future_dataframe(periods=n, freq='D')
            predictions = model.predict(forecast)
            figure = model.plot(predictions)
            plt.title('Cash Flow Projection', fontsize=25)
            return predictions

        def custom_predict(model, data):
            data = data[['date']]
            data.columns = ['ds']
            predictions = model.predict(data)
            return predictions

        def evaluate(y_true, yhat):
            rmse = np.sqrt(mean_squared_error(y_true, yhat))
            r2 = r2_score(y_true, yhat)
            mae = mean_absolute_error(y_true, yhat)
            metrics = [{
                'Root Mean Squared Error' : rmse,
                'R-squared' : r2,
                'Mean Absolute Error' : mae,
            }]

            performance = pd.DataFrame(metrics)
            return performance

        def create_holidays(start_year, end_year):
            holiday_df = []
            holidays = [("New Year's Day", '-01-01'), ("Maundy Thursday", '-04-01'), ("Good Friday", '-04-02'), 
                        ("Araw ng Kagitingan", '-04-09'), ("Labor Day", '-05-01'), ("Independence Day", '-06-12'), 
                        ("National Heroes’ Day", '-08-30'), ("Bonifacio Day", '-11-30'), ("Christmas Day", '-12-25'),
                        ("Rizal Day", '-12-30')]
                        
            for year in range(start_year, end_year+1, 1):
                for h in holidays:
                    holiday_df.append({
                        'ds':str(year)+h[1],
                        'holiday':h[0],
                        'lower_window': 0,
                        'upper_window': 0,
                    })
  
            holiday_df = pd.DataFrame(holiday_df)
            holiday_df['ds'] = pd.to_datetime(holiday_df.ds)
            return holiday_df

        holiday_df = create_holidays(2018, 2022)

        model = train_model(cash_over_time[['date', 'cashflow']], ws=True, ms=True, iw=0.95, holidays=holiday_df)
        predictions = predict(model, 7)
        model_predictions = predict(model, 30)

        st.write(model.plot_components(predictions.reset_index()))

        train_set = cash_over_time[cash_over_time['date'].dt.year != 2021]
        test_set = cash_over_time[cash_over_time['date'].dt.year == 2021]

        model = train_model(train_set[['date', 'cashflow']], ws=True, ms=True, iw=0.95)
        predictions = custom_predict(model, test_set)
        historical_set = historical.set_index('date')
        fig, ax = plt.subplots()
        ax = historical_set['cashflow'].plot(legend=True, label='historical', figsize=(15,5))
        st.pyplot(fig)
        predictions = predictions.set_index('ds')
        fig, ax = plt.subplots()
        ax = predictions['yhat'].plot(legend=True, label='forecast')
        st.pyplot(fig)
        fig, ax = plt.subplots()
        ax = plt.fill_between(predictions.index, predictions['yhat_lower'], predictions['yhat_upper'], color='k', alpha=.15)
        st.pyplot(fig)

        result = model_predictions.reset_index()
        standard_cols = ['date', 'value', 'flag']

        historical = cash_over_time[['date', 'cashflow']]
        historical['flag'] = 'historical'

        forecast = result[-30:][['ds', 'yhat']]
        forecast['flag'] = 'forecast'

        upper = result[-30:][['ds', 'yhat_upper']]
        upper['flag'] = 'upper'

        lower = result[-30:][['ds', 'yhat_lower']]
        lower['flag'] = 'lower'

        historical.columns = standard_cols
        forecast.columns = standard_cols
        upper.columns = standard_cols
        lower.columns = standard_cols

        new_forecast = forecast.append(historical[-1:])
        new_forecast['flag'] = new_forecast['flag'].replace(['historical', 'forecast'])

        new_upper = upper.append(historical[-1:])
        new_upper['flag'] = new_upper['flag'].replace(['historical', 'upper'])

        new_lower = lower.append(historical[-1:])
        new_lower['flag'] = new_lower['flag'].replace(['historical', 'lower'])

        power_bi = pd.concat([new_forecast, new_upper, new_lower, historical], axis=0)
        fig, ax = plt.subplots()
        ax = sns.lineplot(data=power_bi, x='date', y='value', hue='flag')
        st.write(fig)

        # Model Evaluation
        eval_perf = predictions[['yhat']].reset_index()
        eval_perf = eval_perf.merge(test_set, left_on='ds', right_on='date', how='left').drop(['date', 'cash_inflow', 'cash_outflow'], axis=1)
        eval_perf.columns = ['date', 'yhat', 'y']
        st.write(evaluate(eval_perf['y'], eval_perf['yhat']))
        















