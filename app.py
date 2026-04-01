#pip install streamlit 
#pip install pandas 
#pip install sklearn

import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

diabetes_df = pd.read_csv('C:\\Users\\Nitya Karthik A\\DiabetesPrediction\\data\\diabetes.csv')

st.title("Diabetes Checkup")
st.subheader('Training Data')
st.write(diabetes_df.describe())

st.subheader('Visualization')
st.bar_chart(diabetes_df)

x = diabetes_df.drop(['Outcome'], axis=1)
y = diabetes_df.iloc[: , -1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0,100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI',0, 67, 20)
    dpf = st.sidebar.slider('DPF', 0.0,2.4, 0.47)
    age = st.sidebar.slider('Age',21,88, 33 )
    

    user_report =  {
      'Pregnancies':pregnancies,
      'Glucose':glucose,
      'BloodPressure':bp,
      'SkinThickness':skin_thickness,
      'Insulin':insulin,
      'BMI':bmi,
      'DiabetesPedigreeFunction':dpf,
      'Age':age
  }
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report()

# MODEL
rf  = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)


st.subheader("Result"
             )
output = ""
if(user_result[0] == 0):
    output = "Healthy"
else:
    output = "Diabetes"
st.write(output)
st.subheader('Accuracy')
st.write(str(accuracy_score(y_test, rf.predict(x_test))))
