# general libraries
import pickle
import pandas as pd

# model deployment
from flask import Flask
import streamlit as st

st.set_page_config(
    page_title = "Predict the Next AAA Title!",
    # initial_sidebar_state="expanded",
    layout='wide',
    menu_items={
    'About': "### Hi! Thanks for viewing my app!"
    }
)


# read model and holdout data
model = pickle.load(open('xgb.pkl', 'rb'))
X_holdout = pd.read_csv('data/X_holdout.csv', index_col=0)
holdout_transactions = X_holdout.index.to_list()

col1, col2, col3 = st.columns([0.5, 3, 0.5])
with col2:
    st.markdown(f"""<h1 style='text-align: center;'>PREDICT THE AAA MOVIE TITLE</h1>""", unsafe_allow_html=True)
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
                        Welcome to the AAA Movie Predictor!
                    </h1>
                    <p style="font-size: 16px; line-height: 1.5; letter-spacing: 1.5px; color: white;">
                        A simple yet powerful app that predicts whether a movie will achieve AAA status based on its key characteristics. Using a fine-tuned XGBoost machine learning model, the app evaluates important factors such as: Runtime (in minutes), Genres, Actors, Directors, Writers
    
    The app provides clear results: AAA – The movie is predicted to qualify as a top-tier AAA title, Not AAA – The movie is less likely to qualify as an AAA title.
    
    To make the predictions easy to understand, the app also includes a SHAP (SHapley Additive exPlanations) force plot, which explains the influence of each factor on the prediction.
    
    Whether you're exploring movie data or evaluating your own projects, this app offers a practical way to gain insights!
                    </p>
                </div>
                """,
                unsafe_allow_html=True)
    

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
