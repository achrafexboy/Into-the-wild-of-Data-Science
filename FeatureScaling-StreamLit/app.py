import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import os


# Headings
st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("<h1 style='text-align: center; color: Light gray;'>Data Preprocessing Web Application<p style='text-align: center; color:gray;font-size : 15px;'> Into the wild of data Science - Project </p></h1>", unsafe_allow_html=True)
st.sidebar.image("ENSAM-UMI.png", use_column_width=True)
st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Importing DataSets</h3>", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
	            visibility: visible;
	        }
            footer:after {
                content:'Authors :  ACHRAF FAYTOUT / Mohammed AMRANI ALAOUI -- Last-Edit: 03/04/2022 - 14:01'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Enter the path here where all the temporary files will be stored
temp='\\fileDataTempo.csv'
path=os.getcwd()
path=path+temp
  
def get_table_download_link_csv(df):
    try:
        
        """
        Thanks for using Our Web application -- Big thanks to our professor and Mentor : " Mr. Hosni " ❤️
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(
            csv.encode()
        ).decode()  # some strings <-> bytes conversions necessary here
        return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    
    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df



def to_excel(df):
    try:
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer)
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df



def get_table_download_link_xlsx(df):
    try:
        """
        Thanks for using Our Web application -- Big thanks to our professor and Mentor : " Mr. Hosni " ❤️
        """
        val = to_excel(df)
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="dataprep.xlsx">Download xlsx file</a>' # decode b'abc' => abc

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df
    
    
# Feature scaling Technics
  
def z_score(df,column_name):
    try:
        if column_name:
            df['z-score'] = (df[column_name]-df[column_name].mean())/df[column_name].std() #calculating Z-score
            outliers = df[(df['z-score']<-1) | (df['z-score']>1)]   #outliers
            removed_outliers = pd.concat([df, outliers]).drop_duplicates(keep=False)   #dataframe after removal 
            st.dataframe(removed_outliers)
            st.write("Percentile Of Dataset :\n ", df.describe())
            st.write('Number of outliers : {}'.format(outliers.shape[0])) #number of outliers in Given Dataset
            st.info('Size of dataset after outlier removal')
            st.write(removed_outliers.shape)
            st.line_chart(removed_outliers)
            return removed_outliers

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df


def f_ss(df):
    try:
        X = df.select_dtypes(include=np.number)
        mean_X = np.mean(X)
        std_X = np.std(X)
        Xstd = (X - np.mean(X))/np.std(X)
        st.dataframe(Xstd)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xstd.shape)
        st.write('Data Informations :',Xstd.info())
        st.write('Data description : ',Xstd.describe())
        st.line_chart(Xstd)
        return Xstd

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df


def f_mm(df):
    try:
        X = df.select_dtypes(include=np.number)
        min_X = np.min(X)
        max_X = np.max(X)
        Xminmax = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
        st.dataframe(Xminmax)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xminmax.shape)
        st.write('Data Informations :',Xminmax.info())
        st.write('Data description : ',Xminmax.describe())
        st.line_chart(Xminmax)
        return Xminmax
    
    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df


def f_rs(df):
    try:
        X = df.select_dtypes(include=np.number)
        median_X = np.median(X)
        q3=X.quantile(0.75)-X.quantile(0.25)
        Xrs =(X - np.median(X))/q3
        st.dataframe(Xrs)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xrs.shape)
        st.write('Data Informations :',Xrs.info())
        st.write('Data description : ',Xrs.describe())
        st.line_chart(Xrs)
        return Xrs
     
    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df

    
def maxabs(df):
    try:
        X = df.select_dtypes(include=np.number) 
        max_abs_X = np.max(abs(X)) 
        Xmaxabs = X /np.max(abs(X))
        st.dataframe(Xmaxabs)
        st.info("Data to be treated using Feature Scaling : {}".format(list(dict(df.mean()).keys())))
        st.write('Shape of dataframe (Rows, Columns): ',Xmaxabs.shape)
        st.write('Data Informations :',Xmaxabs.info())
        st.write('Data description : ',Xmaxabs.describe())
        st.line_chart(Xmaxabs)
        return Xmaxabs

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df


    
        
    
# feature Scaling options

def fso(df):
    try:
        fs_option=("Standard Scalar","Min Max Scalar", "Max Absolute Scalar" , "Robust Scalar")
        fs_selection=st.sidebar.radio('Choose a Feature Scaling Method',fs_option)

        if fs_selection == 'Standard Scalar':
            st.sidebar.write('you selected Standard Scalar')
            if st.sidebar.button('Process SS'):
                df = pd.read_csv(path)
                df=f_ss(df)
                df.to_csv(path, index=False)
                return df
        elif fs_selection == 'Min Max Scalar':
            st.sidebar.write('you selected min max')
            if st.sidebar.button('Process mm'):
                df = pd.read_csv(path)
                df=f_mm(df)
                df.to_csv(path, index=False)
                return df
        elif fs_selection == 'Max Absolute Scalar':
            st.sidebar.write('You selected max absolute')
            if st.sidebar.button('Process Ma'):
                df = pd.read_csv(path)
                df=maxabs(df)
                df.to_csv(path, index=False)
                return df
        elif fs_selection == 'Robust Scalar':
            st.sidebar.write('You selected Robust Scalar')
            if st.sidebar.button('Process rs'):
                df = pd.read_csv(path)
                df=f_rs(df)
                df.to_csv(path, index=False)
                return df

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df
    
    
def upload_xlsx(uploaded_file): 
    try:
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False) # problem accured when using xlsx format i dont know why ?
            return df
    
    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df


def upload_csv(uploaded_file):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False)
            return df

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df
    
            
# File Upload
def file_upload():
    
    try:
        f_option=('.Xlsx','.Csv')
        f_select=st.sidebar.radio('Choose a file type',f_option)

        if f_select == '.Xlsx':
            uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")
            if uploaded_file:
                if st.sidebar.button('Upload File'):
                    df=upload_xlsx(uploaded_file)
                    return df
        elif f_select == '.Csv':
            uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")
            if uploaded_file:
                if st.sidebar.button('Upload File'):
                    df=upload_csv(uploaded_file)
                    return df
    
    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df
        
    
        

# Data export
def data_export(df):
    try:
        st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Exporting dataSets</h3>", unsafe_allow_html=True)
        fd_option=('.Xlsx','.Csv')
        fd_select=st.sidebar.radio('Choose a file type to download',fd_option)

        if fd_select == '.Csv':
            if st.sidebar.button('Download Csv'):
                df = pd.read_csv(path)
                st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
                return 0

        elif fd_select == '.Xlsx':
            if st.sidebar.button('Download Xlsx'):
                df = pd.read_csv(path)
                st.sidebar.markdown(get_table_download_link_xlsx(df), unsafe_allow_html=True)
                return 0
    
    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df


# Give main options
def main():
    try:
        df=file_upload()
        fso(df)
        data_export(df)

    except Exception as e:
        st.write("WOOOOW :(, ", e.__class__, "Problem occurred.")
        return df
    
main()
        
    
    
    
    
    
    
        
        
    
    
    
    
    
    
