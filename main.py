import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.eda import show_eda
from src.machinelearning import show_machinelearning_analysis
from src.sentimentalanalysis import open_sentimental_analysis_page, process_data
from src.sentimentanalysisdemo import run_demo
from src import design as design

design.showHeader()
# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_page_config(page_title='Title Here',layout="wide")

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

st.sidebar.subheader('Uploaded Data')
table_select = st.sidebar.selectbox(
    label="Show the first 15 rows of the uploaded data?",
    options=['Show', 'Hide']
)

if table_select == 'Show':
    try:
        st.subheader("First 15 Rows")
        st.dataframe(data=df['feedback'].head(50))
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(df.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application.")
else:
    try:
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(df.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application.")

# add a select widget to the side bar
#Data Analysis Selection
st.sidebar.subheader('Data Analysis')
DA_select = st.sidebar.selectbox(
    label="Select the Data Analysis you want to see",
    options=['Charts/EDA', 'Machine Learning', 'Sentiment Analysis', 'Sentiment Analysis Demo']
)

#Charts and EDA Choices
if DA_select =='Charts/EDA':
    try:
        show_eda(df, numeric_columns, non_numeric_columns)
    except Exception as e:
        print(e)

#Machine Learning 
if DA_select == 'Machine Learning':
    try:
        show_machinelearning_analysis(df)
    except Exception as e:
        print(e)

# Sentimental analysis
if DA_select == 'Sentiment Analysis':
    try:
        process_data(df)
    except Exception as e:
        open_sentimental_analysis_page()

# Sentiment Analysis Demo
if DA_select == 'Sentiment Analysis Demo':
    try:
        run_demo(df) 
    except Exception as e:
        pass

design.showFooter()