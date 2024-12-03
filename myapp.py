
#####################################################################################################
#########################################  IN PROGRESS  #############################################
#####################################################################################################
import pickle
import pandas as pd
import joblib
from flask import Flask
import streamlit as st
import shap

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
# model = joblib.load('xgb.pkl')
X_holdout = pd.read_csv('data/X_holdout.csv', index_col=0)
movies = pd.read_csv('data/movies.csv')
X_holdout_id_map = X_holdout.merge(movies, left_index=True, right_index=True, how='left')
holdout_transactions = X_holdout.index.to_list()

# SHAP
explainer = shap.TreeExplainer(model.named_steps['xgbclassifier'])

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
                        PREDICT THE NEXT AAA MOVIE!
                    </h1>
                    <p style="font-size: 16px; line-height: 1.5; letter-spacing: 1.5px; color: white;">
                        Welcome to the AAA Movie Predictor, a simple yet powerful app that predicts whether a movie will achieve AAA status based on its key characteristics. Using a fine-tuned XGBoost machine learning model, the app evaluates important factors such as: Runtime (in minutes), Genres, Actors, Directors, Writers. The app provides clear results: AAA – The movie is predicted to qualify as a top-tier AAA title, Not AAA – The movie is less likely to qualify as an AAA title.
                        <br><br>To make the predictions easy to understand, the app also includes a SHAP (SHapley Additive exPlanations) force plot, which explains the influence of each factor on the prediction. Whether you're exploring movie data or evaluating your own projects, this app offers a practical way to gain insights!
                    </p>
                </div>
                """,
                unsafe_allow_html=True)

    st.divider()
    st.header("Craft and Predict Movie Idea!")
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
    



    st.divider()
    st.header("Predicting Unseen Movie Data")
    

    #adding a selectbox
    choice = st.selectbox(
        "Choose Movie to Predict:",
        options = X_holdout_id_map["primaryTitle"].to_list(),
        placeholder = "Type or Search the Movie. See (?) for more details.",
        index = None,
        help = "These movies are treated as unseen data of the Predictive Model.")

    
    
    def predict_if_AAA(transaction_id):
        transaction = X_holdout.loc[transaction_id].values.reshape(1, -1)
        prediction_num = model.predict(transaction)[0]
        pred_map = {1: 'AAA', 0: 'Not AAA'}
        prediction = pred_map[prediction_num]
        return prediction

    if not choice == 0:
        
        movie_index_label = X_holdout_id_map[X_holdout_id_map['primaryTitle'] == choice].index[0]
        if st.button("Predict"):
            output = predict_if_AAA(movie_index_label)
        
            if output == 'AAA':
                st.success('Movie could be the next AAA title!')
            elif output == 'Not AAA':
                st.error('Not AAA')
    
            IMDB_Rating = X_holdout_id_map["averageRating"].loc[movie_index_label]
            st.markdown(f"IMDB Rating = {IMDB_Rating}")

        
            # shap_values = explainer.shap_values(X_holdout, check_additivity=False)
            # instance_index = X_holdout.index.get_loc(movie_index_label)
            # shap.force_plot(
            #     explainer.expected_value, 
            #     shap_values[instance_index],  
            #     X_holdout.loc[movie_index_label], 
            #     feature_names=X_holdout.columns.tolist()
            # )
