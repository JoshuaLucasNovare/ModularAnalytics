import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_eda(df, numeric_columns, non_numeric_columns):
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
















