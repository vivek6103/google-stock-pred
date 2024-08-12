import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Ignore the ComplexWarning


# Load the trained model from the pickle file
with open('model12.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit web page title
st.title("ML Model Deployment with Streamlit")

# Get user input for model features
st.header("Enter Input Features for Prediction")

# Example feature input fields (customize these based on your model's features)
feature_1 = st.number_input('date', value=0)
feature_2 = st.number_input('close', value=0)
feature_3 = st.number_input('high', value=0)
feature_4 = st.number_input('low', value=0)
feature_5 = st.number_input('open', value=0)
feature_6 = st.number_input('volume', value=0)
feature_7 = st.number_input('adjHigh', value=0)
feature_8 = st.number_input('adjLow', value=0)
feature_9 = st.number_input('adjOpen', value=0)
feature_10 = st.number_input('adjVolume', value=0)


# Arrange the inputs in the same order as expected by the model
new_df=pd.DataFrame({
    'date':[feature_1],
    'close':[feature_2],
    'high':[feature_3],
    'low':[feature_4],
    'open':[feature_5],
    'volume':[feature_6],
    'adjHigh':[feature_7],
    'adjLow':[feature_8],
    'adjOpen':[feature_9],
    'adjVolume':[feature_10]
})

# input_features = np.array([[0,18,18,17,17,498,18,17,17,498]])

# Button to make a prediction
if st.button('Predict'):
    prediction = model.predict(new_df)
    st.write(f"Prediction: {prediction}")

# Additional information or results can be displayed here
st.write("Adjust the feature inputs to see how the prediction changes.")
