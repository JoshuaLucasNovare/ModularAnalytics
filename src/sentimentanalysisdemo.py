import os

import findspark
findspark.init(os.environ['SPARK_HOME'])
import joblib
import time
import re
import logging
import string
from datetime import datetime
import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from translator import Translator
from .twitter.twitter_search import get_tweets
from textblob.translate import Translator
from textblob import TextBlob
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
            time.sleep(1.0)
            sentence = sentence.strip()
            print(f'Sentence: {sentence}')
            en_blob = Translator()
            translated.append(str(en_blob.translate(sentence, from_lang='tl', to_lang='en')))
            # en_blob = TextBlob(sentence)
            # translated.append(en_blob.translate(from_lang='tl', to='en'))
            time.sleep(1.0)
        except Exception as e:
            translated.append(sentence)
            print(f'[TRANSLATION ERROR] {e}')

    return translated


def stop_remover(x, stop_list):
    lst = []
    for i in x:
        if i.lower() not in stop_list:
            lst.append(i.lower())
    return lst


def run_demo(df):
    st.title("Sentiment Analysis")

    try:
        st.sidebar.subheader("Data Sentiment Analysis")
        feedback_column = st.sidebar.selectbox(
            label="Comment Columns", 
            options=df.columns
        )
        print(f'Feedback column: {feedback_column}\ttype: {type(feedback_column)}')
        
        if st.sidebar.button("Process Data"):
            df['sentiment'] = ''
            feedbacks = df[feedback_column].values
            f = st.empty()
            t = st.empty()
            s = st.empty()
            for index, row in df.iterrows(): # feedback in feedbacks:
                f.markdown(f'Feedback: {row[feedback_column]}')
                time.sleep(1.0)
                translated = translate_to_eng(row[feedback_column])
                t.markdown(f'English Transation: {translated}')
                # time.sleep(1.0)
                vect = vectorizer.transform(translated)
                sentiment = pipeline.predict(vect)[0]
                s.markdown(f'Sentiment: {sentiment}')
                time.sleep(1.0)
                f.markdown('')
                t.markdown('')
                s.markdown('')
                df.at[index, 'sentiment'] = sentiment
                time.sleep(1.0)
            st.dataframe(data=df)

            # Cleaning the original reviews for wordcloud output
            stopwords = pd.read_csv(f'{os.getcwd()}/src/data/stopwords.txt', sep=' ', header=None)
            stopwords.columns = ['words']
            custom = ['sana', 'po', 'yung', 'mas', 'ma', 'kasi', 'ninyo', 'kayo', 'nya', 'pag', 'naman', 'lang', 'no', 'comment']
            stop_list = stopwords['words'].values.tolist()  + custom
        
            # Applying Word Tokenization
            df['word_tokenized'] = df[feedback_column].apply(word_tokenize)
            df[feedback_column] = df['word_tokenized'].apply(lambda x: stop_remover(x, stop_list))
            df[feedback_column] = df[feedback_column].apply(lambda x: ' '.join(x))
            df = df.drop(['word_tokenized'], axis=1)

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


def open_demo_sentimental_analysis_page():
    try:
        st.sidebar.subheader("Twitter Search")
        keywords = st.sidebar.text_input("Keyword")

        if keywords:
            tweets = get_tweets(keywords)
            tweets = tweets.drop_duplicates(subset='tweet', keep='last')
            # print(type(tweets))
            print(f"Tweets: {tweets}")
            st.write(tweets)
            tweets.to_csv(f'LandbankTwitter{datetime.today()}.csv', index=False)
            run_demo(tweets)
    except Exception as e:
        print(f"Error Except: {e}")















































