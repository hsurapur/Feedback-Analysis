import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Function to load custom stopwords from a file
def load_custom_stopwords(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().splitlines())

# Load tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load label mapping
with open('label_mapping.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the pre-trained model
model = load_model('gru_rnn.keras')

# Load custom stopwords
stop_words = load_custom_stopwords('english_stopwords.txt')

# Step 2: Helper Functions
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    tokens = text.split()
    # Remove stopwords without lemmatization
    tokens = [word for word in tokens if word not in stop_words]  
    return ' '.join(tokens)

# Streamlit app
st.title('Feedback Sentiment')
st.write('Enter a message to find the sentiment of it')

# User input
user_input = st.text_area('Feedback')

if st.button('Find Sentiment'):
    # Preprocess the input
    preprocessed_input = preprocess_text(user_input)
    input_sequence = tokenizer.texts_to_sequences([preprocessed_input])
    input_padded = pad_sequences(input_sequence, maxlen=60, padding='post')

    # Make predictions
    predictions = model.predict(input_padded)
    predicted_class = np.argmax(predictions, axis=-1)

    # Inverse label mapping
    inverse_label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    predicted_label = inverse_label_mapping[predicted_class[0]]

    # Display sentiment based on predicted label
    if predicted_label == 'good':
        st.write("Sentiment: Positive")
    elif predicted_label == 'bad':
        st.write("Sentiment: Negative")
    else:
        st.write("Sentiment: Neutral")
else:
    st.write('Please enter a feedback.')