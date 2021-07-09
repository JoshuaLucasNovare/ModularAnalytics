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
from textblob.translate import Translator
from time import sleep
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

def remove_punctuations(text):
    """
    Removes Punctuations from a string
    """
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
        text = text.lower()
    return text

def concat_reasons(cols, data):
    df = data.copy()
    df = df.dropna(subset=cols, thresh=1)
    df['concat_reasons'] = df[cols[0]].str.cat(
        df[cols[1]], sep=". ", na_rep='').str.lstrip('.').str.strip()
    return df

def translate_to_eng(paragraph):
    paragraph = str(paragraph).strip().split('.')
    translated = []

    for sentence in paragraph:
        try:
            sleep(1.0)
            sentence = sentence.strip()
            en_blob = Translator()
            translated.append(str(en_blob.translate(sentence, from_lang='tl', to_lang='en')))
            sleep(1.0)
        except Exception as e:
            print(e)

    return translated

def text_translation(df, cols):
    df = concat_reasons(cols, df)
    df = df[df['concat_reasons'].apply(remove_punctuations) != 'areas for improvement']
    df = df[df['concat_reasons'].apply(remove_punctuations) != 'no comment']
    df = df[df['concat_reasons'].apply(remove_punctuations) != 'compliment']
    # df.rename({'feedback_mode':'mode', 'office':'office location'}, axis = 1, inplace = True)
    # df = df[['date_created', 'gender','mode','office location', 'concat_reasons']]
    df['concat_reasons'] = df['concat_reasons'].replace(r'\s\.', '', regex = True)
    df['concat_reasons'] = df['concat_reasons'].str.replace('NA.', '')

    final_data = df.copy()
    final_data['concat_reasons'] = final_data['concat_reasons'].replace(r'\s\.', '', regex = True)
    final_data['translated'] = final_data['concat_reasons'].apply(translate_to_eng)

    return final_data

def perform_sentimental_analysis(df):
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
    stopwords = pd.read_csv('data/stopwords.txt', sep=' ', header=None)
    stopwords.columns = ['words']
    custom = ['sana', 'po', 'yung', 'mas', 'ma', 'kasi', 'ninyo', 'kayo', 'nya', 'pag', 'naman', 'lang', 'no', 'comment']
    stop_list = stopwords['words'].values.tolist()  + custom

    # Applying Word Tokenization
    df['word_tokenized'] = df['original'].apply(word_tokenize)
    df['original'] = df['word_tokenized'].apply(lambda x: stop_remover(x, stop_list))
    df['original2'] = df['original'].apply(lambda x: ' '.join(x))
    df_to_powerbi = df[['original2', 'sentiment']]
    df_to_powerbi.rename({'original2': 'text'}, axis = 1, inplace = True)
    df_to_powerbi = df_to_powerbi[df_to_powerbi['text'] != '']
    st.write(df_to_powerbi.head(15))
    
    return df_to_powerbi

def process_data(df):
    st.title("Sentimental Analysis")

    try:
        print("Showing wordcloud")
        st.sidebar.subheader("Data Sentimental Analysis")
        comment_columns = st.sidebar.multiselect(
            label="Comment Columns", 
            options=df.columns
        )
        
        if st.sidebar.button("Process Data"):
            df_translated = text_translation(df, comment_columns)
            df_translated = df_translated[['concat_reasons', 'translated']]
            # st.write(df_translated)
            show_wordcloud(df_translated)
        # show_wordcloud(df)
    except Exception as e:
        print(e)

def show_wordcloud(df):
    st.title("Sentimental Analysis")
    df_to_powerbi = perform_sentimental_analysis(df)

    try:
        print("Showing wordcloud")
        # col_select = st.sidebar.selectbox(
        #     label="Select Column",
        #     options=df_to_powerbi.columns.tolist()
        # )
        # target_col = st.sidebar.selectbox(
        #     label="Target Column",
        #     options=df_to_powerbi.columns.tolist()
        # )

        df_cat = {}
        comb = {}
        long_str = {}
        wordcloud = {}
        categories = list(set(df_to_powerbi["sentiment"].tolist()))

        colors = ["#BF0A30", "#002868"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)

        for cat in categories:
            df_cat[cat] = df_to_powerbi[df_to_powerbi["sentiment"] == cat]
            comb[cat] = df_cat[cat]["text"].values.tolist()
            long_str[cat] = ' '.join(comb[cat])
            wordcloud[cat] = WordCloud(background_color="white", colormap=cmap, width=1000, 
                     height=300, max_font_size=500, relative_scaling=0.3, 
                     min_font_size=5)
            wordcloud[cat].generate(long_str[cat])

            st.subheader(f"Category {cat}")
            st.image(image=wordcloud[cat].to_image(), caption=f"Category {cat}")
    except Exception as e:
        print(e) # sentiment















