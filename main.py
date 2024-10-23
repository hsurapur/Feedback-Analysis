import numpy as np
import pandas as pd
import re
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the pre-trained model
model = load_model('gru_rnn.keras')

# Load custom stopwords
def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().splitlines())

stopwords = load_stopwords('english_stopwords.txt')

# Helper function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Handle negations more effectively
    negations = ["not", "no", "never", "none", "n't"]
    words = text.split()
    
    # Create a flag to indicate if the current word is negated
    negation_flag = False
    processed_words = []

    for word in words:
        if word in negations:
            negation_flag = True
        else:
            if negation_flag:
                processed_words.append("not_" + word)  # Prefix 'not_' to the word
                negation_flag = False  # Reset the flag after the first word
            else:
                processed_words.append(word)

    text = ' '.join(processed_words)

    # Remove URLs, mentions, hashtags, punctuation, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

# Streamlit app
st.title('Feedback Sentiment Analysis')
st.write('Enter a message to find the sentiment of it')

# User input
user_input = st.text_area('Feedback')

if st.button('Find Sentiment'):
    # Preprocess the input
    preprocessed_input = preprocess_text(user_input)
    input_sequence = tokenizer.texts_to_sequences([preprocessed_input])
    input_padded = pad_sequences(input_sequence, maxlen=60, padding='pre')

    # Make predictions
    predictions = model.predict(input_padded)
    predicted_class = (predictions < 0.5).astype(int)
    predicted_label = 'Positive' if predicted_class[0][0] == 1 else 'Negative'
    predicted_probability = predictions[0][0]

    # Display results
    st.write(f"Sentiment: {predicted_label}")

else:
    st.write('Please enter feedback.')