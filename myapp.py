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

st.markdown(f"""<h1 style='text-align: center;'>MAKE THE AAA MOVIE TITLE</h1>""", unsafe_allow_html=True)
st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #009688, #3F51B5);
                padding: 40px;
                border-radius: 10px;
                text-align: center;
                font-family: Arial, sans-serif;
                color: white;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
            ">
                <h1 style="font-size: 28px; margin-bottom: 10px; font-weight: bold; letter-spacing: 2px; color: white; text-transform: capitalize;">
                    Ready To Navigate Your Data Science Journey?
                </h1>
                <p style="font-size: 18px; line-height: 1.5; letter-spacing: 1.5px; color: white;">
                    <strong>Learn and Be Guided with Confidence!</strong> EskwelApps is here to guide you every step of the way. Whether you're exploring the perfect learning path, seeking detailed program insights, or looking for a personalized assessment, we’ve got everything you need to thrive. <strong><br><br>Unlock Tools and Resources!</strong> Once enrolled, dive into our comprehensive course outline, get your questions answered with our bootcamp assistant, and easily set up your environment with our installation guide. Let EskwelApps support you in your Eskwelabs' data science journey.
                </p>
            </div>
            """,
            unsafe_allow_html=True)


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
