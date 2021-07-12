import streamlit.components.v1 as components
import streamlit as st
from PIL import Image

def showHeader():
    bvm = Image.open(r"images/logo.png")
    st.image(bvm)

def showFooter():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}     
            footer:after {
            content:'9/F MDi Corporate Center, 39th Street, cor. 10th Ave., Taguig, Metro Manila, Philippines, 1634'; 
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)