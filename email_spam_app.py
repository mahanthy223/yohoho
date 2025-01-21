import streamlit as st
import pickle

# Load the trained model and vectorizer
with open('Email_Spam_Detection.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('Email_Spam_Detection.pkl', 'rb') as vec_file:
    feature_extraction = pickle.load(vec_file)

# Streamlit application title
st.title('Email/SMS Spam Classifier')

# Input text from the user
input_sms = st.text_input("Enter the Message")

# Preprocess, Vectorize, and Predict
if input_sms:
    input_sms = [input_sms]  # Wrap input in a list (even if it's just one message)

    # Vectorize the input using the pre-fitted vectorizer (no need to fit again)
    input_sms_vectorized = feature_extraction.transform(input_sms)

    # Predict using the loaded model
    prediction = loaded_model.predict(input_sms_vectorized)

    # Display the result
    if prediction[0] == 0:
        st.header("Ham Mail")  # 0 means not spam (ham)
    else:
        st.header("Spam Mail")  # 1 means spam
