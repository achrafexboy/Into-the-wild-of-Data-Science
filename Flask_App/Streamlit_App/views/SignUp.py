import streamlit as st

# passlib, hashlib, bcrypt, scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# DB management functions 
def create_usertable(c):
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, email TEXT, password TEXT)')

def add_userdata(c, conn, username, email,password):
    c.execute('INSERT INTO userstable(username, email, password) VALUES (?,?,?)',(username,email, password))
    conn.commit()

def load_view(c, conn):
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
    st.title("Create New Account")
    new_user = st.text_input("Username",key='1')
    new_email = st.text_input("E-mail adresse",key='2')
    new_password = st.text_input("Password",type='password',key='3')

    if st.button("Sign Up"):
        create_usertable(c)
        add_userdata(c, conn, new_user, new_email, make_hashes(new_password))
        st.success("You have successfully created a valid Account")
        st.info("Go to Login Menu to login")

