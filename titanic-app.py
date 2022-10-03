import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.header("Welcome to our Tiranic Streamlit Web App")
st.subheader("""The page is divied in two categories:
             1. PowerBI report on Titanic dataset
             2. Data preprocessing and predictions""")

options = st.selectbox('Please Select', ['PowerBI', 'PreProcessing &predictions'])


if options == 'PowerBI':
    st.markdown("link copied from powerbi web service", unsafe_allow_html=True)

else:
    url = 'https://raw.githubusercontent.com/YJPark0421/Streamlit-POWERBI/main/titanic.csv'
    df = pd.read_csv(url, index_col=0).reset_index()
    df = df.dropna()
    st.write(df.head())
       
    def user_input_features():
        pclass = st.sidebar.selectbox('Pclass',[0,1])
        sex = st.sidebar.selectbox('Sex',[0,1])
        age = st.sidebar.slider('Age', 0.42, 80.00, 31.0)
        sibsp = st.sidebar.slider('SibSp', 0, 5, 2)
        parch = st.sidebar.slider('Parch', 0, 6, 2)
        fare = st.sidebar.slider('Fare', 0.0, 513.0, 2.0)
        embarked = st.sidebar.slider('Embarked', 1, 3, 2)
        data = {'pclass': pclass, 'sex': sex, 'age': age
                , 'sibsp': sibsp, 'parch': parch, 'fare': fare
                , 'embarked': embarked}
        features = pd.DataFrame(data, index=[0])
        return features
    st.sidebar.slider('User Input Parameters')
    df1 = user_input_features()
    
    st.write(df.info())
    st.write(df.describe())
    
    st.subheader('User Input Parameters')
    st.write(df1)
    y = df['Survived']
    X = df.iloc[:, 1:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Reads in saved classification model
load_clf = pickle.load(open('titanic_clf.pkl', 'rb'))

prediction = load_clf.predict(df1)
prediction_proba = load_clf.predict_proba(df1)

st.subheader('Prediction Probability')
st.write(prediction_proba)
