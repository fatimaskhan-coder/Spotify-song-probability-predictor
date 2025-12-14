{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b4427f77-3685-428a-9bb6-e58282dfe58e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5af442fd-7d08-4b88-bca8-46597b613469",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load saved model, scaler, and one-hot columns\n",
    "model = joblib.load(\"spotify_rf_model.pkl\")\n",
    "scaler = joblib.load(\"spotify_scaler.pkl\")\n",
    "encoded_columns = joblib.load(\"spotify_encoded_columns.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "bc2f1559-ec73-4eba-9849-00137448fedc",
   "metadata": {},
   "outputs": [],
   "source": [
    "danceability = st.slider(\"Danceability %\", 0, 100, 50)\n",
    "energy = st.slider(\"Energy %\", 0, 100, 50)\n",
    "valence = st.slider(\"Valence %\", 0, 100, 50)\n",
    "acousticness = st.slider(\"Acousticness %\", 0, 100, 50)\n",
    "artist_count = st.number_input(\"Artist Count\", 1, 5, 1)\n",
    "key = st.selectbox(\"Key\", ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'])\n",
    "mode = st.selectbox(\"Mode\", ['Major', 'Minor'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "a963f66a-c48b-4e6f-a47a-a81e7452f0a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create input dataframe\n",
    "input_df = pd.DataFrame([[danceability, energy, valence, acousticness, artist_count]],\n",
    "                        columns=['danceability_%','energy_%','valence_%','acousticness_%','artist_count'])\n",
    "\n",
    "# One-hot encode key and mode\n",
    "for col in encoded_columns:\n",
    "    input_df[col] = 1 if col.split('_')[1] in [key, mode] else 0\n",
    "\n",
    "# Scale features\n",
    "input_scaled = scaler.transform(input_df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e736cbb-7c18-4fac-8fb8-ecb88c03fb95",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "475f6223-9677-4bf9-ace8-adf54210540d",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = model.predict(input_scaled)[0]\n",
    "probability = model.predict_proba(input_scaled)[0][1]\n",
    "\n",
    "st.write(f\"Prediction: {'Popular' if prediction==1 else 'Not Popular'}\")\n",
    "st.write(f\"Probability of being popular: {probability*100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "b1d247df-62f0-41d3-ae50-9e119f51d501",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Usage: streamlit run [OPTIONS] TARGET [ARGS]...\n",
      "Try 'streamlit run --help' for help.\n",
      "\n",
      "Error: Invalid value: File does not exist: spotify_app.py\n"
     ]
    }
   ],
   "source": [
    "!streamlit run spotify_app.py\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "be85b74d-4869-43ac-a6bf-ebf99f84b4da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Streamlit title\n",
    "st.title(\"Spotify Song Popularity Predictor\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "c2a33d6b-83f4-44d9-90dc-0aa0aa77f0cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['spotify_scaler.pkl']"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(scaler, \"spotify_scaler.pkl\")  # saves scaler\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "4acc3bf6-e88f-47d8-a12d-0e81c659310d",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'encoded_cat' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[33], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m joblib\u001b[38;5;241m.\u001b[39mdump(\u001b[43mencoded_cat\u001b[49m\u001b[38;5;241m.\u001b[39mcolumns, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mspotify_encoded_columns.pkl\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'encoded_cat' is not defined"
     ]
    }
   ],
   "source": [
    "joblib.dump(encoded_cat.columns, \"spotify_encoded_columns.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "9b24a83e-570d-45e3-b69c-af69a1f5d217",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load model\n",
    "model = joblib.load(\"spotify_rf_model.pkl\")\n",
    "\n",
    "# Load scaler\n",
    "scaler = joblib.load(\"spotify_scaler.pkl\")\n",
    "\n",
    "# Load one-hot columns\n",
    "encoded_columns = joblib.load(\"spotify_encoded_columns.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e95d094b-080e-4bc8-9365-cd9294ea4008",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
