import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud


def show_wordcloud(df):
    print("Hello, sentimental analysis")
    try:
        st.sidebar.subheader("Data Sentimental Analysis")
        col_select = st.sidebar.selectbox(
            label="Select Column",
            options=df.columns.tolist()
        )
        target_col = st.sidebar.selectbox(
            label="Target Column",
            options=df.columns.tolist()
        )

        df_cat = {}
        comb = {}
        long_str = {}
        wordcloud = {}
        categories = list(set(df[target_col].tolist()))

        colors = ["#BF0A30", "#002868"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)

        for cat in categories:
            df_cat[cat] = df[df[target_col] == cat]
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















