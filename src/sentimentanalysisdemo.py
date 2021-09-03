import os

import findspark
findspark.init(os.environ['SPARK_HOME'])
import joblib
import time
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
            df['sentiment'] = ''
            feedbacks = df['feedback'].values
            f = st.empty()
            t = st.empty()
            s = st.empty()
            for index, row in df.iterrows(): # feedback in feedbacks:
                f.markdown(f'Feedback: {row["feedback"]}')
                # time.sleep(1.0)
                translated = translate_to_eng(row["feedback"])
                t.markdown(f'English Transation: {translated}')
                # time.sleep(1.0)
                vect = vectorizer.transform(translated)
                sentiment = pipeline.predict(vect)[0]
                s.markdown(f'Sentiment: {sentiment}')
                # time.sleep(1.0)
                f.markdown('')
                t.markdown('')
                s.markdown('')
                df.at[index, 'sentiment'] = sentiment
                # time.sleep(1.0)
            st.dataframe(data=df)
            show_wordcloud(df)
    except Exception as e:
        print(e)

def show_wordcloud(df):

    try:
        df_cat = {}
        comb = {}
        long_str = {}
        wordcloud = {}
        categories = list(set(df["sentiment"].tolist()))

        colors = ["#BF0A30", "#002868"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)

        for cat in categories:
            df_cat[cat] = df[df["sentiment"] == cat]
            comb[cat] = df_cat[cat]["feedback"].values.tolist()
            long_str[cat] = ' '.join(comb[cat])
            wordcloud[cat] = WordCloud(background_color="white", colormap=cmap, width=1000, 
                     height=300, max_font_size=500, relative_scaling=0.3, 
                     min_font_size=5)
            wordcloud[cat].generate(long_str[cat])

            st.subheader(f"Category {cat}")
            st.image(image=wordcloud[cat].to_image(), caption=f"Category {cat}")
    except Exception as e:
        print(f'Exception: {e}')














































