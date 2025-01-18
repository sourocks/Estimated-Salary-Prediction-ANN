import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model("regression_model.h5")
# Load the trained model and preprocessing files
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit App
st.title('Customer Estimated Salary Prediction (Regression)')
st.markdown("### Predict the customer's estimated salary based on provided details")

# Collapsible section for user inputs
with st.expander("Enter Customer Details"):
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, value=30)
    balance = st.number_input('Balance', min_value=0.0, format="%0.2f", value=50000.0)
    credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, step=1, value=600)
    tenure = st.slider('Tenure (years)', 0, 10, value=5)
    num_of_products = st.slider('Number of Products', 1, 4, value=2)
    has_cr_card = st.selectbox('Has Credit Card', ["No", "Yes"])
    is_active_member = st.selectbox('Is Active Member', ["No", "Yes"])
    exited = st.selectbox('Exited', [0, 1])

    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == "Yes" else 0],
        'IsActiveMember': [1 if is_active_member == "Yes" else 0],
        'Exited': [exited],
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine input data with encoded geography
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the estimated salary
    prediction = model.predict(input_data_scaled)
    estimated_salary = float(prediction[0])

    # Display the results
    st.markdown("### Prediction Results")
    st.write(f'Estimated Salary: ${estimated_salary:,.2f}')

    # Progress bar for visualization
    min_salary, max_salary = 0, 200000  # Adjust range as needed
    normalized_value = (estimated_salary - min_salary) / (max_salary - min_salary)
    normalized_value = max(0, min(normalized_value, 1))  # Clamp between 0 and 1
    st.progress(normalized_value)
