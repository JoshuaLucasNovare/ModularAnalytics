import os

import findspark
findspark.init(os.environ['SPARK_HOME'])
import joblib
import re
import logging
import string
import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from translator import Translator
from .twitter.twitter_search import get_tweets
from textblob.translate import Translator
from time import sleep
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
from ast import literal_eval
from nltk.tokenize import word_tokenize
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

data_dir = 'data'
models_dir = 'models'

src = 'src'
data = 'SentimentAnalysisDemoData.csv'
model = 'multinomialnb.pkl'

pipeline = joblib.load(f'{src}/{models_dir}/{model}')
vectorizer = joblib.load(f'{src}/{models_dir}/vectorizer.pkl')

def translate_to_eng(paragraph):
    paragraph = str(paragraph).strip().split('.')
    translated = []

    for sentence in paragraph:
        try:
            sentence = sentence.strip()
            en_blob = Translator()
            translated.append(str(en_blob.translate(sentence, from_lang='tl', to_lang='en')))
        except Exception as e:
            print(e)

    return translated

def run_demo(df):
    st.title("Sentiment Analysis")

    try:
        st.sidebar.subheader("Data Sentiment Analysis")
        comment_columns = st.sidebar.multiselect(
            label="Comment Columns", 
            options=df.columns
        )
        
        if st.sidebar.button("Process Data"):
            feedbacks = df['feedback'].values
            for feedback in feedbacks:
            	st.text(f'Feedback: {feedback}')
            	translated = translate_to_eng(feedback)
            	st.text(f'English Transation: {translated[0]}')
            	vect = vectorizer.transform(translated)
            	st.text(f'Sentiment: {pipeline.predict(vect)[0]}')
            	st.markdown("""---""")
    except Exception as e:
        print(e)














































