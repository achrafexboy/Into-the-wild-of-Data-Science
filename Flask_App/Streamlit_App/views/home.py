import streamlit as st
import requests
from streamlit_lottie import st_lottie

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding1 = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_coding2 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_lw4fol0h.json")
lottie_coding3 = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_ri7tiddf.json")
lottie_coding4 = load_lottieurl("https://assets8.lottiefiles.com/private_files/lf30_zd4ppbmb.json")

def load_view():
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            with open("assets/styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                st.subheader("Into The Wild Of Data Science :zap:")
                st.title("What is Data Preprocessing?")
                st.write(
                    "Data preprocessing is a data mining technique that involves transforming raw data into an understandable format because Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends."
                )
        with right_column:
            st_lottie(lottie_coding1, height=300, key="coding")

    with st.container():
        left_column, right_column = st.columns(2)
        with right_column:
            with open("assets/styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                st.title("Handling Missing Values")
                st.write(
                    "Handling missing values is an important step in data cleaning that can impact model validity and reliability."
                )
                st.write(
                    "In this part of the project we will handle the missing values in the dataset using different type of methods like Mean imputation and Median imputation and Mode imputation (Categorical features) and also culmns o rows deleting and Other methods that you can try."
                )
        with left_column:
            st_lottie(lottie_coding2, height=400, key="missingValues")

    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            with open("assets/styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                st.title("Feature Scaling")
                st.write(
                    "Feature scaling is a method used to normalize the range of independent variables or features of data. we do feature scaling to make the flow of gradient descent smooth and helps algorithms quickly reach the minima of the cost function. "
                )
                st.write(
                    "In this part of the project we will apply Feature scaling to the features of the dataset using different type of methods like z_score, Min-Max, Max-Absolute scaling, Robust Scaling and Other methods that you can try."
                )
        with right_column:
            st_lottie(lottie_coding3, height=400, key="featureScaling")

    with st.container():
        left_column, right_column = st.columns(2)
        with right_column:
            with open("assets/styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                st.title("Data Reduction")
                st.write(
                    "Data reduction is the process of reducing the amount of capacity required to store data. Data reduction can increase storage efficiency and reduce costs. Storage vendors will often describe storage capacity in terms of raw capacity and effective capacity, which refers to data after the reduction."
                )
                st.write(
                    "In this part of the project we will apply some data reduction techniques to our dataset using PCA (Principal component Analysis) and factor analysis, you can try those methods in our app."
                )
        with left_column:
            st_lottie(lottie_coding4, height=400, key="dataReduction")
