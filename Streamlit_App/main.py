from codecs import utf_16_be_decode
import streamlit as st
import utils as utl
import subprocess
from PIL import Image
from streamlit_lottie import st_lottie
from views import about, contactUs, home, LogIn, SignUp, App

# DB connexion
import sqlite3
conn = sqlite3.connect('users.db')
c = conn.cursor()
img = Image.open('assets/images/DS.png')
st.set_page_config(layout="wide", page_title='Into The Wild of DS', page_icon=img)
utl.inject_custom_css()
utl.navbar_component()

# Go = False it's not working , this method od course
# Style footer
st.markdown("""<style>
            #MainMenu {visibility: hidden;}
            footer {
                visibility: hidden;
            }
            footer:after {
                content : "• Into the wild of data science © 2022";
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
                font-family : "Sofia Pro" ;
                font-size : 15px;
            }
            .block-container{
                padding-top : 0px;
            }
            </style>
            """, unsafe_allow_html= True)
def navigation():
    route = utl.get_current_route()
    if route == "home":
        home.load_view()
    elif route == "LogIn":
        LogIn.load_view(c)
    elif route == "SignUp":
        SignUp.load_view(c, conn)
    elif route == "contactUs":
        contactUs.load_view()
    elif route == "about":
        about.load_view()
    elif route == "App" :
        # subprocess.call("streamlit run App.py", shell=True) # Runing a script inside our script main.py
        App.load_view()
    else :
        home.load_view()

navigation()

# What I did in this 5 hours :
    # I create a boolean 'Go'
    # then I asked for verification from LogIn route
    # if the user is in our DB then the LogIn view will return True value
    # and then WE will have access to the route App otherwise => home section ...