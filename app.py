# email_spam_classifier_app.py

# Importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the pre-trained model and vectorizer
model = pickle.load(open('model1.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))

# Initialize NLTK requirements
nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

# Function to preprocess and transform the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(words)

# Streamlit app code
st.title("Email/SMS Spam Classifier")

# Input box for user to enter email or SMS text
input_text = st.text_area("Enter the email or SMS text you want to classify:")

# Process the input and predict using the model
if st.button("Classify"):
    if input_text.strip() == "":
        st.write("Please enter some text for classification.")
    else:
        # Transform and vectorize the input text
        transformed_text = transform_text(input_text)
        vectorized_text = tfidf.transform([transformed_text]).toarray()

        # Predict and output the result
        prediction = model.predict(vectorized_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        
        # Display the result
        st.header(f"Prediction: {result}")
