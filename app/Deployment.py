from Tkinter import bitmap 
import streamlit as st
import pandas as pd
import streamlit as st
from scipy.stats import boxcox
import os
import numpy as np
import json
import requests

from streamlit_lottie import st_lottie

def load_lottifile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieurl(url : str):

    r=  requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Model 
def app():
    #Backend Section

    df = pd.read_csv("app\insurance.csv")

    # Variable yang digunakan
    categorical_columns = ['sex','children', 'smoker', 'region']
    df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
    
    y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)


    ## Log Transformasi
    df_encode['charges'] = np.log(df_encode['charges'])


    # Proses Training 
    from sklearn.model_selection import train_test_split
    X = df_encode.drop('charges',axis=1) # Independet variable
    y = df_encode['charges'] # dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)


    # Langkah 1: add x0 =1 to dataset
    X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
    X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

    # Langkah 2: build model
    theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,y_train)) 

    # Parameter untuk model Linear regression
    parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
    columns = ['intersect:x_0=1'] + list(X.columns.values)
    parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})

    # Library Scikit Learn module
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

    #Parameter
    sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
    parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))


    y_pred_sk = lin_reg.predict(X_test)

    # Data Frame untuk inputan

    n_df= pd.DataFrame(columns=['age','sex','bmi', 'children', 'smoker', 'region'])

    #Frontend
    st.subheader("Silahkan lengkapi beberapa informasi dibawah ini !")

    #Inputan User

    #Umur
    age = st.slider("Berapa Usia Anda ?" ,1 , 100, 20)
    st.write(" Usia Anda ", age , "Tahun")

    #Jenis kelamin 
    sex = st.selectbox(
     'Pilih Jenis Kelamin',
     ('Pria', 'Wanita'))

    st.write('Jenis Kelamin:', sex)

    # Ubah Jenis Kelamin Menjadi Angka Sesuai dengan Data
    if sex == 'Pria':
        OHE_male= 1
    else:
        OHE_male= 0	

    #BMI 
    bmi = st.number_input('Silahkan Masukkan BMI Anda ( Body Mass Index )')
    st.write('BMI Anda : ', bmi)

    #Anak 
    child = st.slider("Berapa Jumlah Anak Yang Anda Miliki ? ?" ,1 , 5, 2)
    st.write(" Kamu Mempunyai ", child , "Anak")

    OHE_1 = 0
    OHE_2= 0
    OHE_3= 0
    OHE_4= 0
    OHE_5 = 0

    if child == 1:
        OHE_1= 1
    elif child == 2:
        OHE_2= 1
    elif child == 3:
        OHE_3= 1
    elif child == 4:
        OHE_4= 1
    else:
        OHE_5= 1

    #Perokok 
    smoke = st.radio(
     "Apakah Anda Seorang Perokok ?",
     ('Ya', 'Tidak'))

    OHE_yes =0

    if smoke == 'Ya':
        st.write('Anda Seorang Perokok.')
        OHE_yes= 1
    else:
        st.write("Anda Bukan Seorang Perokok.")
    
    #Daerah Regional 

    region = st.radio(
     "Pilih Daerah Regional",
     ('Indonesia Bagian Timur', 'Indonesia Bagian Tengah', 'Indonesia Bagian Barat', 'Luar Negeri'))

    OHE_southwest= 0
    OHE_southeast= 0
    OHE_northwest= 0

    if region == 'Indonesia Bagian Timur':
        st.write('Daerah Regional Anda : Indonesia Bagian Timur.')
        OHE_southwest=1

    elif region == 'Indonesia Bagian Tengah':
        st.write('Daerah Regional Anda : Indonesia Bagian Tengah.')
        OHE_southeast = 1
    elif region == 'Indonesia Bagian Barat':
        st.write('Daerah Regional Anda : Indonesia Bagian Barat.')
        OHE_northwest= 1
    else:
        st.write("Anda berada di Luar Negeri.")

    
    st.markdown("---")
    
    st.subheader(" Klik Tombol dibawah untuk menghitung prediksi :")


    # Membuat Prediksi dengan inputan user

    #n_df.insert(age,sex,bmi, child, smoke, region)

    n_df= pd.DataFrame(columns=['age', 'bmi', 'OHE_male', 'OHE_1', 'OHE_2', 'OHE_3', 'OHE_4', 'OHE_5',
       'OHE_yes', 'OHE_northwest', 'OHE_southeast', 'OHE_southwest'])

    data = [ [age, bmi, OHE_male, OHE_1, OHE_2, OHE_3, OHE_4, OHE_5,
       OHE_yes, OHE_northwest, OHE_southeast, OHE_southwest]]
    n_df = pd.DataFrame( data , columns = ['age', 'bmi', 'OHE_male', 'OHE_1', 'OHE_2', 'OHE_3', 'OHE_4', 'OHE_5',
       'OHE_yes', 'OHE_northwest', 'OHE_southeast', 'OHE_southwest'])

    pred= lin_reg.predict(n_df)

    charge = np.exp(pred)

    #Hasil Perhitungan convert ke Rupiah
    note = (charge[0])*15159

    
    if st.button('Hitung Prediksi Biaya'):
        st.write('Prediksi Biaya : Rp ', note )
        lottie_hello= load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_de909vf3.json")
        st_lottie(
            lottie_hello,
            speed=1,
            reverse= False,
            loop=True,
            quality= "low",
            height= None,
            width= None,
            key=12,
        )

# side bar






    


   
