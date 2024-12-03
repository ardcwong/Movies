
#####################################################################################################
#########################################  IN PROGRESS  #############################################
#####################################################################################################


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
movies = pd.read_csv('data/movies.csv')
holdout_transactions = X_holdout.index.to_list()


col1, col2, col3 = st.columns([0.5, 3, 0.5])
with col2:
    # st.markdown(f"""<h1 style='text-align: center;'>PREDICT THE AAA MOVIE TITLE</h1>""", unsafe_allow_html=True)
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
                    <h1 style="font-size: 40px; margin-bottom: 10px; font-weight: bold; letter-spacing: 2px; color: white; text-transform: capitalize;">
                        PREDICT THE AAA MOVIE TITLE
                    </h1>
                    <p style="font-size: 16px; line-height: 1.5; letter-spacing: 1.5px; color: white;">
                        Welcome to the AAA Movie Predictor, a simple yet powerful app that predicts whether a movie will achieve AAA status based on its key characteristics. Using a fine-tuned XGBoost machine learning model, the app evaluates important factors such as: Runtime (in minutes), Genres, Actors, Directors, Writers. The app provides clear results: AAA – The movie is predicted to qualify as a top-tier AAA title, Not AAA – The movie is less likely to qualify as an AAA title.
                        <br><br>To make the predictions easy to understand, the app also includes a SHAP (SHapley Additive exPlanations) force plot, which explains the influence of each factor on the prediction. Whether you're exploring movie data or evaluating your own projects, this app offers a practical way to gain insights!
                    </p>
                </div>
                """,
                unsafe_allow_html=True)

    a, b = st.columns([1,1])
    with a:
        title = st.text_input("Movie Title")
        genre = st.multiselect("Genre",['Action', 'Adult', 'Adventure', 'Animation',
           'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
           'Fantasy', 'Game-Show', 'History', 'Horror', 'Music', 'Musical',
           'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Sport',
           'Talk-Show', 'Thriller', 'War', 'Western'])
        runtime = st.number_input("Runtime (Minutes)", min_value = 0, max_value = 180)
        

    with b:
        director = st.multiselect("Director/s",["D1","D2"]) #upload directors list
        writer = st.multiselect("Writer/s", ["D1","D2"])
        actor = st.multiselect("Actor/s", ["D1","D2"])
    




    

    #adding a selectbox
    choice = st.selectbox(
        "Select Transaction Number:",
        options = movies.loc(holdout_transactions)["primaryTitle"])
    
    
    def predict_if_AAA(transaction_id):
        transaction = X_holdout.loc[transaction_id].values.reshape(1, -1)
        prediction_num = model.predict(transaction)[0]
        pred_map = {1: 'AAA', 0: 'Not AAA'}
        prediction = pred_map[prediction_num]
        return prediction
    
    if st.button("Predict"):
        output = predict_if_AAA(choice.index.get_loc(movies))
    
        if output == 'Fraud':
            st.sucess('AAA')
        elif output == 'Not Fraud':
            st.error('Not AAA')
