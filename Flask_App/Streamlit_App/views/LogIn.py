import streamlit as st
import pandas as pd

# passlib, hashlib, bcrypt, scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB management functions 
def create_usertable(c):
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, email TEXT, password TEXT)')

def login_user(c, username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data

def view_all_users(c):
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def load_view(c):
    editCss = """
        <style>
            .row-widget{
                width : 500px;
                margin : 0 auto;
            }
            .edgvbvh9{
                width : 100px;
                transform : translateX(-50%);
                margin-left: 50%;
                margin-top : 20px;
            }
        </style>
    """
    st.markdown(editCss, unsafe_allow_html=True)    
    st.title('Login to your DS account')
    username = st.text_input("User Name")
    password = st.text_input("Password",type='password')
    if st.button("Login"):
        # if password == '12345':
        create_usertable(c)
        hashed_pswd = make_hashes(password)

        result = login_user(c, username,check_hashes(password,hashed_pswd))
        if result:

            st.success("Logged In as {}".format(username))
            st.write('Now, Go to "Application" ... ')
            return True
        else:
            st.warning("Incorrect Username/Password")


# Class names of the first input and label : element-container css-1nvpywl e1tzin5v3
# class names of the second input and lable : element-container css-1nvpywl e1tzin5v3
# login button classes : element-container css-1nvpywl e1tzin5v3