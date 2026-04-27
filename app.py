import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load preprocessors
scaler = joblib.load('scaler.joblib')
imputer = joblib.load('imputer.joblib')

st.title("🧠 ASD Prediction App (Logistic & XGBoost Only)")

# Model selection dropdown
model_choice = st.selectbox("Select Model", ["Logistic Regression", "XGBoost"])

# Load selected model
if model_choice == "Logistic Regression":
    model = joblib.load('logistic_model.joblib')
else:
    model = joblib.load('xgb_model.joblib')

st.write("Fill the details below:")

# Inputs
age = st.number_input("Age", 1, 120, 25)
gender = st.selectbox("Gender", ['m', 'f'])
ethnicity = st.selectbox("Ethnicity", [
    'White-European', 'Latino', 'Others', 'Black', 'Asian',
    'Middle Eastern ', 'Pasifika', 'South Asian', 'Hispanic', 'Turkish'
])
relation = st.selectbox("Relation", [
    'Self', 'Parent', 'Relative', 'Health care professional', 'Others'
])
country = st.selectbox("Country", [
    'United States', 'India', 'United Kingdom', 'Canada', 'Brazil', 'UAE',
    'New Zealand', 'Australia', 'Pakistan', 'Saudi Arabia', 'Others'
])

# A1 - A10
st.subheader("Screening Questions (0 = No, 1 = Yes)")
a_scores = {}
for i in range(1, 11):
    a_scores[f"A{i}_Score"] = st.selectbox(f"A{i} Score", [0, 1], key=f"a{i}")

jundice = st.selectbox("Had Jaundice at birth?", [0, 1])
austim = st.selectbox("Family member with autism?", [0, 1])
used_app_before = st.selectbox("Used screening app before?", [0, 1])

# Age group function
def convert_age_group(age):
    if age < 4: return 'Toddler'
    elif age < 12: return 'Kid'
    elif age < 18: return 'Teenager'
    elif age < 40: return 'Young'
    else: return 'Senior'

age_group = convert_age_group(age)

# Label encoding helper
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data

# Predict button
if st.button("Predict"):

    # sum_score and ind calculation
    sum_score = sum(a_scores.values())
    ind = austim + used_app_before + jundice

    # Create input dataframe (19 features)
    new_data = pd.DataFrame([{
        'age': np.log(age),
        'gender': gender,
        'ethnicity': ethnicity,
        'relation': relation,
        'country_of_res': country,
        'result': sum_score,
        'ageGroup': age_group,
        'sum_score': sum_score,
        'ind': ind,
        'A1_Score': a_scores['A1_Score'],
        'A2_Score': a_scores['A2_Score'],
        'A3_Score': a_scores['A3_Score'],
        'A4_Score': a_scores['A4_Score'],
        'A5_Score': a_scores['A5_Score'],
        'A6_Score': a_scores['A6_Score'],
        'A7_Score': a_scores['A7_Score'],
        'A8_Score': a_scores['A8_Score'],
        'A9_Score': a_scores['A9_Score'],
        'A10_Score': a_scores['A10_Score'],
    }])

    # Encode
    new_data = encode_labels(new_data)

    # Impute
    new_data_imputed = imputer.transform(new_data)

    # Scale
    new_data_scaled = scaler.transform(new_data_imputed)

    # Predict
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)[0][1] * 100

    # Show result
    if prediction[0] == 1:
        st.error(f"🟥 ASD Positive (Risk: {probability:.2f}%)")
    else:
        st.success(f"🟩 ASD Negative (Risk: {probability:.2f}%)")
