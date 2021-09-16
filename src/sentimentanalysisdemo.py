import os

import findspark
findspark.init(os.environ['SPARK_HOME'])
import joblib
import tweepy
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
from .twitter.utils import get_language, remove_enter, translate_to_eng
from google.cloud import translate_v2 as translate
from .twitter.twitter_search import get_tweets, tw_oauth
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
            df['translated'] = ''
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
                df.at[index, 'translated'] = translated
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
            df.to_csv(f'SentimentAnalysis_{datetime.today()}.csv')

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


def tw_search():
    try:
        st.sidebar.subheader("Twitter Search")
        keywords = st.sidebar.text_input("Keyword")
        api = tw_oauth()

        tweet_list = {
            'source': [],
            'tweet': [],
            'date_posted': [],
            'translated': [],
            'sentiment': []
        }

        f = st.empty()
        t = st.empty()
        s = st.empty()

        if keywords:
            tweets = tweepy.Cursor(api.search, q=keywords)

            for tweet in tweets.items():
                row = remove_enter(tweet.text)

                if row not in tweet_list['tweet']:
                    f.markdown(f'Tweet: {row}')
                    time.sleep(2.0)

                    translated = translate_to_eng(row)
                    t.markdown(f'English Transation: {translated}')
                    time.sleep(2.0)

                    vect = vectorizer.transform(translated)
                    sentiment = pipeline.predict(vect)[0]
                    s.markdown(f'Sentiment: {sentiment}')

                    time.sleep(3.0)
                    f.markdown('')
                    t.markdown('')
                    s.markdown('')

                    tweet_list['source'].append('Twitter')
                    tweet_list['tweet'].append(row)
                    tweet_list['date_posted'].append(tweet.created_at)
                    tweet_list['translated'].append(translated)
                    tweet_list['sentiment'].append(sentiment)


            tweet_df = pd.DataFrame(tweet_list)
            tweet_df.to_csv(f'LandbankTwitter{datetime.today()}.csv', index=False)
    except Exception as e:
        print(f"Error Except: {e}")















































