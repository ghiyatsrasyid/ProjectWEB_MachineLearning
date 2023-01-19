from ast import Mod
import streamlit as st
from multiapp import MultiApp
from app import dashboard, Deployment
import json

import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie


def load_lottifile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url : str):

    r=  requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.markdown("<h1 style='text-align: center; '>Prediksi Biaya Kesehatan Perokok dengan Linear Regression</h1>", unsafe_allow_html=True)

app = MultiApp()


app.add_app("Hitung Prediksi",Deployment.app)
app.add_app("Visualisasi Data",dashboard.app)


st.sidebar.title("Machine Learning 3 TI E")
st.sidebar.text("")
st.sidebar.image("img\logo-psti.png", use_column_width=True)
st.sidebar.text("")
st.sidebar.title("Nama Kelompok :")
st.sidebar.text("1. Muhammad Ghiyats Rasyid")
st.sidebar.text("2. Muhammad Nayyul Habibie")
st.sidebar.text("3. Nabila Aurora Destiani")
st.sidebar.text("")
st.sidebar.title("Apa Itu Linear Regression")
st.sidebar.markdown("""Regresi linear adalah teknik analisis data yang memprediksi nilai data yang tidak 
diketahui dengan menggunakan nilai data lain yang terkait dan diketahui. Secara matematis memodelkan variabel 
yang tidak diketahui atau tergantung dan variabel yang dikenal atau independen sebagai persamaan linier.""")

app.run()