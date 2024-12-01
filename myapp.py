# general libraries
import pickle
import pandas as pd

# model deployment
from flask import Flask
import streamlit as st

# read model and holdout data
model = pickle.load(open('xgb.pkl', 'rb'))
X_holdout = pd.read_csv('data/X_holdout.csv', index_col=0)
holdout_transactions = X_holdout.index.to_list()

st.title("Transaction Fraud Detection")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> Credit Card Fraud Detection ML App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

#adding a selectbox
choice = st.selectbox(
    "Select Transaction Number:",
    options = holdout_transactions)


def predict_if_fraud(transaction_id):
    transaction = X_holdout.loc[transaction_id].values.reshape(1, -1)
    prediction_num = model.predict(transaction)[0]
    pred_map = {1: 'Fraud', 0: 'Not Fraud'}
    prediction = pred_map[prediction_num]
    return prediction

if st.button("Predict"):
    output = predict_if_fraud(choice)

    if output == 'Fraud':
        st.error('This transaction may be FRAUDULENT', icon="🚨")
    elif output == 'Not Fraud':
        st.success('This transaction is approved!', icon="✅")
