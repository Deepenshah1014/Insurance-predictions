Skip to main content
Google Classroom
Classroom
AND-NOV2024-DSAI-1
Home
Calendar
Enrolled
To do
M
Machine Learning BIA
A
AND-NOV2024-DSAI-1
Archived classes
Settings
Stream
Classwork
People
Stream was updated
AND-NOV2024-DSAI-1
Upcoming
Woohoo, no work due in soon!

Announce something to your class

Post by Tirth Mehta
Tirth Mehta
Created 17:3117:31
Artifacts

model_rest.joblib
Binary

model_young.joblib
Binary

scaler_rest.joblib
Binary

scaler_young.joblib
Binary

Add class comment…


Post by Tirth Mehta
Tirth Mehta
Created 17:3117:31
Health insurance

main.py
Text

prediction_helper.py
Text

requirements.txt
Text

Add class comment…


Post by Anand Solomon Phulpagar
Anand Solomon Phulpagar
Created 17:0917:09
scikit-learn==1.6.1
streamlit==1.43.2
xgboost==3.0.0
joblib==1.4.2
pandas==2.2.3
numpy==2.2.4

Add class comment…


Announcement: 'Code'
Pranav Mulye
Created 16:4316:43
Code

New folder.zip
Compressed archive

Add class comment…


Post by Deepen Shah
Deepen Shah
Created 23 Mar23 Mar
for pranav sir to help

premium_prediction_ young.zip
Compressed archive

Add class comment…


Post by Yashvi Shah
Yashvi Shah
Created 23 Mar23 Mar
import streamlit as st
from prediction_helper import predict

# Set page config for a bright and fun UI
st.set_page_config(page_title="🎉 Health Insurance Predictor", layout="wide")

# Custom CSS for colorful styling
st.markdown("""
    <style>
        /* Background Gradient */
        .stApp {
            background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
        }

        /* Title */
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #4A148C;
            margin-bottom: 20px;
        }

        /* Section headings */
        .subheading {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #D81B60;
            margin-top: 10px;
        }

        /* Container styling */
        .stContainer {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 4px 12px rgba(0, 0, 0, 0.2);
        }

        /* Sliders */
        .stSlider {
            color: #7B1FA2;
        }

        /* Select and input field styling */
        .stTextInput, .stNumberInput, .stSelectbox {
            background: #F3E5F5 !important;
            color: black !important;
            border-radius: 10px;
            border: 2px solid #D81B60;
        }

        /* Button Styling */
        .stButton>button {
            background: linear-gradient(45deg, #FF4081, #7B1FA2) !important;
            color: white !important;
            font-size: 18px;
            padding: 12px;
            border-radius: 8px;
            width: 100%;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #7B1FA2, #FF4081) !important;
        }

        /* Prediction Box */
        .prediction-box {
            background: linear-gradient(45deg, #4CAF50, #2E7D32);
            color: white;
            padding: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# 🎉 Title
st.markdown('<p class="title">🎉 Health Insurance Predictor</p>', unsafe_allow_html=True)

# Define dropdown categories
categorical_options = {
    'Gender': ['👨 Male', '👩 Female'],
    'Marital Status': ['💍 Married', '💔 Unmarried'],
    'BMI Category': ['✅ Normal', '⚠️ Obesity', '⚖️ Overweight', '🔹 Underweight'],
    'Smoking Status': ['🚭 No Smoking', '🔥 Regular', '⚠️ Occasional'],
    'Employment Status': ['💼 Salaried', '🚀 Self-Employed', '💻 Freelancer'],
    'Region': ['🌍 Northwest', '🏝️ Southeast', '🗺️ Northeast', '🏔️ Southwest'],
    'Medical History': [
        '🟢 No Disease', '🩸 Diabetes', '⚠️ High Blood Pressure', '🩸 Diabetes & High BP',
        '🦋 Thyroid', '❤️ Heart Disease', '⚠️ BP & Heart Disease', '🩸 Diabetes & Thyroid',
        '🩸 Diabetes & Heart Disease'
    ],
    'Insurance Plan': ['🥉 Bronze', '🥈 Silver', '🥇 Gold']
}

# 📊 Dashboard Container
with st.container():
    st.markdown('<div class="stContainer">', unsafe_allow_html=True)

    # 📋 User Details Heading
    st.markdown('<p class="subheading">📋 Enter Your Details</p>', unsafe_allow_html=True)

    # 🎚️ Sliders for age & income
    age = st.slider('🎂 Age', min_value=18, max_value=100, value=25)
    income_lakhs = st.slider('💰 Income in Lakhs', min_value=0, max_value=200, value=10)

    # 🏷️ Arrange Inputs in Three-Column Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('⚧️ Gender', categorical_options['Gender'])
        smoking_status = st.selectbox('🚬 Smoking Status', categorical_options['Smoking Status'])
        bmi_category = st.selectbox('⚖️ BMI Category', categorical_options['BMI Category'])
        employment_status = st.selectbox('💼 Employment Status', categorical_options['Employment Status'])

    with col2:
        number_of_dependants = st.number_input('👨‍👩‍👦 Number of Dependants', min_value=0, step=1, max_value=20)
        marital_status = st.selectbox('💍 Marital Status', categorical_options['Marital Status'])
        region = st.selectbox('🌎 Region', categorical_options['Region'])
        genetical_risk = st.slider('🧬 Genetical Risk', min_value=0, max_value=5, value=2)

    with col3:
        insurance_plan = st.selectbox('📜 Insurance Plan', categorical_options['Insurance Plan'])
        medical_history = st.selectbox('🏥 Medical History', categorical_options['Medical History'])

    # Close Container Styling
    st.markdown('</div>', unsafe_allow_html=True)

# 📊 Collect user inputs in a dictionary
input_dict = {
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Genetical Risk': genetical_risk,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

# 🔮 Prediction Section
st.markdown('<p class="subheading">🔮 Prediction</p>', unsafe_allow_html=True)

# Prediction Button
if st.button('🎯 Predict Insurance Cost'):
    prediction = predict(input_dict)
    st.markdown(f'<p class="prediction-box">💰 Predicted Health Insurance Cost: {prediction} INR</p>', unsafe_allow_html=True)

Add class comment…


Announcement: 'Latest Project Code.'
Pranav Mulye
Created 23 Mar23 Mar
Latest Project Code.

Project_Code.zip
Compressed archive

Add class comment…


Announcement: 'code'
Pranav Mulye
Created 23 Mar23 Mar
code

prediction_helper.py
Text

main.py
Text

Add class comment…


Announcement: 'sns.histplot(results_df['diff_pct'],…'
Pranav Mulye
Created 23 Mar23 Mar
sns.histplot(results_df['diff_pct'], kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Diff PCT')
plt.ylabel('Frequency')
plt.show()

Add class comment…


Announcement: 'numeric_features = ['age',…'
Pranav Mulye
Created 23 Mar23 Mar
numeric_features = ['age', 'income_lakhs', 'number_of_dependants', 'genetical_risk']

fig, axes = plt.subplots(1, len(numeric_features), figsize=(18, 6))  # Adjust figure size as necessary

for ax, column in zip(axes, numeric_features):
    sns.scatterplot(x=df2[column], y=df2['annual_premium_amount'], ax=ax)
    ax.set_title(f'{column} vs. Annual Premium Amount')
    ax.set_xlabel(column)
    ax.set_ylabel('Annual Premium Amount')

plt.tight_layout()  # Adjust layout
plt.show()

Add class comment…

# codebasics ML course: codebasics.io, all rights reserverd

import pandas as pd
import joblib

model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score

def preprocess_input(input_dict):
    # Define the expected columns and initialize the DataFrame with zeros
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    # df.fillna(0, inplace=True)

    # Manually assign values for each categorical input based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':  # Correct key usage with case sensitivity
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':  # Correct key usage with case sensitivity
            df['age'] = value
        elif key == 'Number of Dependants':  # Correct key usage with case sensitivity
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':  # Correct key usage with case sensitivity
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # Assuming the 'normalized_risk_score' needs to be calculated based on the 'age'
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df

def handle_scaling(age, df):
    # scale age and income_lakhs column
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])
prediction_helper.py
Displaying requirements.txt.