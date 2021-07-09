import joblib
import re
import logging
import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
from ast import literal_eval
from nltk.tokenize import word_tokenize


logging.basicConfig(level=print)

def fix_translated(original, translated):
    """
    Appends the columns original and translated if the conditions are satisfied.
    
    """
    f = '. '
    if translated == '[]':
        f = original
    else:
        translated = literal_eval(str(translated))
        f = f.join(translated)

    return f

def stop_remover(x, stop_list):
    lst = []
    for i in x:
        if i.lower() not in stop_list:
            lst.append(i.lower())
    return lst    

def punctuation(string):
    # punctuation marks
    punctuations = '''"#$%&\()*+',‘“”’-/:;<=>_@[\\]^`{|}~?!.'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, " ")
    return string.lower()

def punctuation2(string):
    # punctuation marks
    punctuations = '''.'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, " ")
    text_no_doublespace = re.sub('\s+', ' ', string).strip()
    return text_no_doublespace

def clean_text(text):
    # remove numbers
    text_nonum = re.sub(r'\d+', '', text)
#     # remove punctuations and convert characters to lower case
#     text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
#     # substitute multiple whitespace with single whitespace
#     # Also, removes leading and trailing whitespaces
#     text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_nonum

def show_wordcloud(df):
    print("Hello, sentimental analysis")
    df['clean_translated'] = df.apply(lambda x: fix_translated(
        x['concat_reasons'], x['translated']), axis=1).str.lower()
    
    type_comments = ['no comment', 'compliment', 'areas for improvement',
                'products/services to offer in the future', 'good service',
                'complaint', 'products/services to offer', 'product/service to offer',
                 'lower interest rate']
    df.dropna(subset = ['concat_reasons', 'clean_translated'], inplace = True)
    df['clean_translated'] = df['clean_translated'].apply(clean_text)
    df['clean_translated'] = df['clean_translated'].apply(punctuation2)
    df = df[~df['clean_translated'].str.strip().isin(type_comments)]
    df.rename({'clean_translated':'review'}, axis=1, inplace=True)
    reviews = df['review'].values
    print("Got reviews")

    pipeline = joblib.load('models/pipeline.pkl')
    predictions = pipeline.predict(reviews)
    print(f"Predictions: {predictions}")
    df['sentiment'] = predictions
    df.rename({'concat_reasons':'original'}, axis=1, inplace=True)
    # df.drop('review', axis=1, inplace=True)

    # Cleaning the original reviews for wordcloud output
    test_translated = pd.read_csv('data/translated.csv')
    print(f"test_translated: {test_translated}")
    stopwords = pd.read_csv('data/stopwords.txt', sep=' ', header=None)
    stopwords.columns = ['words']
    custom = ['sana', 'po', 'yung', 'mas', 'ma', 'kasi', 'ninyo', 'kayo', 'nya', 'pag', 'naman', 'lang', 'no', 'comment']
    stop_list = stopwords['words'].values.tolist()  + custom

    # Applying Word Tokenization
    df['word_tokenized'] = df['original'].apply(word_tokenize)
    df['original'] = df['word_tokenized'].apply(lambda x: stop_remover(x, stop_list))
    df['original2'] = df['original'].apply(lambda x: ' '.join(x))
    df_to_powerbi = df[['year', 'gender', 'mode', 'office location', 'original2', 'sentiment']]
    df_to_powerbi.rename({'original2': 'text'}, axis = 1, inplace = True)
    df_to_powerbi = df_to_powerbi[df_to_powerbi['text'] != '']
    st.write(df_to_powerbi.head(15))
    df_to_powerbi.to_csv('results.csv', index=False)

    try:
        print("Showing wordcloud")
        st.sidebar.subheader("Data Sentimental Analysis")
        col_select = st.sidebar.selectbox(
            label="Select Column",
            options=df_to_powerbi.columns.tolist()
        )
        target_col = st.sidebar.selectbox(
            label="Target Column",
            options=df_to_powerbi.columns.tolist()
        )

        df_cat = {}
        comb = {}
        long_str = {}
        wordcloud = {}
        categories = list(set(df_to_powerbi[target_col].tolist()))

        colors = ["#BF0A30", "#002868"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)

        for cat in categories:
            df_cat[cat] = df_to_powerbi[df_to_powerbi[target_col] == cat]
            comb[cat] = df_cat[cat][col_select].values.tolist()
            long_str[cat] = ' '.join(comb[cat])
            wordcloud[cat] = WordCloud(background_color="white", colormap=cmap, width=1000, 
                     height=300, max_font_size=500, relative_scaling=0.3, 
                     min_font_size=5)
            wordcloud[cat].generate(long_str[cat])

            st.subheader(f"Category {cat}")
            st.image(image=wordcloud[cat].to_image(), caption=f"Category {cat}")
    except Exception as e:
        print(e)















