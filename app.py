import streamlit as st
import pandas as pd
import os 
# import profiling capability

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
# ML staff
from pycaret.classification import setup,compare_models,pull,save_model
st.balloons()
with st.sidebar:
    st.image("https://media.istockphoto.com/id/1371766825/photo/big-data-network-abstract-concept.jpg?b=1&s=170667a&w=0&k=20&c=0C8TyRQTYkX-q_jm0pzc-MjZ7of_CcCisqPF4VMX_ug=")
    st.title("AutoStreamML")
    choices = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This appliacation allows you to build an automated ML pipline using Streamlit Pandas Profiling")
    st.warning("Please enter the data with no nan value")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

if choices =="Upload":
    st.title("Upload Your Data Of Modeling")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df=pd.read_csv(file,index_col=None)
        df = df.dropna()
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)
        pass

if choices == "Profiling":
    st.title("Automatic Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
if choices == "ML":
    st.title("Machine Learning go BRR***")
    target = st.selectbox("Select your target",df.columns)
    if st.button("Train model"):

        setup(df,target=target)
        setup_df = pull()
        st.info("This is the MML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')

if choices == "Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("Download the Model",f,"trained_model.pkl")

