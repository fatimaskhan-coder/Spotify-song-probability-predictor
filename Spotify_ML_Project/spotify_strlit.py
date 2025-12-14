import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and one-hot columns
model = joblib.load("spotify_rf_model.pkl")
scaler = joblib.load("spotify_scaler.pkl")
encoded_columns = joblib.load("spotify_encoded_columns.pkl")

# Streamlit UI
st.title("Spotify Song Popularity Predictor 🎵")
st.write("Predict whether a song will be popular based on audio features.")

# User inputs
danceability = st.slider("Danceability %", 0, 100, 50)
energy = st.slider("Energy %", 0, 100, 50)
valence = st.slider("Valence %", 0, 100, 50)
acousticness = st.slider("Acousticness %", 0, 100, 50)
artist_count = st.number_input("Artist Count", 1, 5, 1)
key = st.selectbox("Key", ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'])
mode = st.selectbox("Mode", ['Major', 'Minor'])

# Create input dataframe
input_df = pd.DataFrame([[danceability, energy, valence, acousticness, artist_count]],
                        columns=['danceability_%','energy_%','valence_%','acousticness_%','artist_count'])

# One-hot encode key and mode
for col in encoded_columns:
    input_df[col] = 1 if col.split('_')[1] in [key, mode] else 0

# Scale features
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.write(f"Prediction: {'Popular' if prediction==1 else 'Not Popular'}")
st.write(f"Probability of being popular: {probability*100:.2f}%")
