from email.mime import image
import streamlit as st
from PIL import Image

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

#For dataset
import pandas as pd
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization

# Convert gambar menjadi bitimage
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def app():
    st.write(" ### Unggah Data Anda ")
   
    uploaded_file = st.file_uploader("Pilih file")
    if uploaded_file is not None:
        
        #read csv
        df =pd.read_csv(uploaded_file)

        st.write(" ### Library yang digunakan")

        st.code("""
        # Librairies
        import pandas  as pd #Data manipulation
        import numpy as np #Data manipulation
        import matplotlib.pyplot as plt #Visualization
        import seaborn as sns #Visualization 
        """)
        st.markdown(" Data Random")
        st.write(df.head())

        st.write("")
        st.write("")
        st.write("")

        st.subheader("Visualisasi Data ")

        st.markdown("Visualisasi Data menggunakan Seaborn library dengan BMI dan Biaya sebagai variabel independen ")
        
        st.code(""" # Visualisasi
            sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)
            plt.xlabel('Body Mass Index$(kg/m^2)$: as Independent variable')
            plt.ylabel('Insurance Charges: as Dependent variable')
            plt.title('Charge Vs BMI')
        """)

        #figure
        #fig = plt.figure(figsize =(4, 4))
        #sns.lineplot(x='bmi',y='charges',data=df )
        #st.pyplot(fig)

        f= plt.figure(figsize=(8,8))
        sns.lineplot(x="age", y="charges",
             hue="sex",
             data=df)
        plt.title("Biaya vs Umur(LinePlot)")
        st.write(f)
        
        st.write("")
        st.write("")
        st.write("")

        st.subheader("Exploratory Data Analysis (EDA) ")
        st.code("df.describe()")
        st.write(df.describe())

        st.write("")
        st.write("")
        st.write("")

        st.subheader("Korelasi antara features(HeatMap)")
        st.code("""
            # Korelasi heatmap
            corr = df.corr()
            sns.heatmap(corr, cmap = 'Wistia', annot= True);
        
        """)

        # correlation fig
        corr = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, ax=ax , cmap = 'Wistia', annot= True)
        st.write(fig)

        st.subheader(" Distribusi Pertanggungan Biaya ")
        st.code(
            """ 
            f= plt.figure(figsize=(12,4))
            ax=f.add_subplot(121)
            sns.distplot(df['charges'],bins=50,color='r',ax=ax)
            ax.set_title('Distribusi Pertanggungan Biaya')
            ax=f.add_subplot(122)
            sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
            ax.set_title('Distribusi Pertanggungan Biaya dalam $log$ scale')
            ax.set_xscale('log'); 
            """
        )
        f= plt.figure(figsize=(12,4))
        ax=f.add_subplot(121)
        sns.distplot(df['charges'],bins=50,color='r',ax=ax)
        ax.set_title('Distribusi Pertanggungan Biaya')
        ax=f.add_subplot(122)
        sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
        ax.set_title('Distribution of insurance charges in $log$ sacle')
        ax.set_xscale('log')
        st.write(f)

        st.subheader("Biaya vs Usia & Biaya vs BMI (Scatter Plot) ")
        st.code(""" 
        f = plt.figure(figsize=(14,6))
        ax = f.add_subplot(121)
        sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker',ax=ax)
        ax.set_title('Scatter plot of Charges vs age')
        ax = f.add_subplot(122)
        sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
        ax.set_title('Scatter plot of Charges vs bmi')
        """ )

        f = plt.figure(figsize=(14,6))
        ax = f.add_subplot(121)
        sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker',ax=ax)
        ax.set_title('Scatter plot of Charges vs age')
        ax = f.add_subplot(122)
        sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
        ax.set_title('Scatter plot of Charges vs bmi')
        st.write(f)

        




    else:
        st.warning(" Silahkan Upload Data dengan format .CSV atau .XLS ")
    

    
    

    
