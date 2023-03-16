import streamlit as st
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe1.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Old Car Price Predictor")


car = st.selectbox('Car Name', df['car_name'].unique())

distance = st.number_input('KMS Driven in kms')

fuel = st.selectbox('Fuel Type', df['fuel_type'].unique())

transmission = st.selectbox('Transmission', df['transmission'].unique())

ownership=st.selectbox('Ownership', df['ownership'].unique())

manufacture = st.number_input('Registration Year')

engine = st.number_input('Engine Displacement in CC')

seats = st.selectbox('Seats',[2,4,5,6,7,8])

if st.button('Pridict Price'):
    query = np.array([car,distance,fuel, transmission,ownership,manufacture,engine,seats])
    query=query.reshape([1,8])
    st.title("The predicted price of this car is: " + str(int(np.exp(pipe.predict(query)[0]))))