import streamlit as st
import pandas as pd
from joblib import load
import dill

with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')
print(my_feature_dict.keys())

def predict_churn(data):
    prediction = model.predict(data)
    return prediction

st.title("Employee Attrition Prediction App")
st.subheader("Based on Employee Dataset")

st.subheader("Categorical Features")
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals={}
for i, col in enumerate(categorical_input.get('Column Name').values()):
    categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i],key=col)

st.subheader("Numerical Features")
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals={}
for col in  numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.number_input(col, min_value=0, value=0, step=1, key=col)

input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_data = pd.DataFrame.from_dict(input_data,orient='index').T

print(input_data)

if st.button('Predict'):
    prediction = predict_churn(input_data)[0]
    translation_dict = {1: "Expected", 0: "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    st.write(f'The Prediction is **{prediction}**, Hence customer is **{prediction_translate}** to churn.')