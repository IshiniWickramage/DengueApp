import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt


st.set_page_config(page_title="Dengue Predictor", layout="wide")

# LOAD MODELS 
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

numerical_features = [
    'Year', 'Month', 'Latitude', 'Longitude',
    'Elevation', 'Temp_avg', 'Precipitation_avg',
    'Humidity_avg', 'Lagged_Cases'
]

district_cols = encoder.get_feature_names_out(['District'])
features = numerical_features + list(district_cols)

st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #121212;
}

/* All normal text */
html, body, [class*="css"]  {
    color: #FFFFFF;
}

/* Title styling */
h1, h2, h3, h4 {
    color: #00BFFF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E1E1E;
}

/* Input boxes (safe way) */
div[data-baseweb="input"] > div {
    background-color: #2A2A2A !important;
    color: white !important;
    border-radius: 8px;
}

/* Input text */
div[data-baseweb="input"] input {
    color: white !important;
}

/* Selectbox dropdown */
div[data-baseweb="select"] > div {
    background-color: #2A2A2A !important;
    color: white !important;
}

/* Slider text */
.stSlider label {
    color: white !important;
}

/* Buttons */
.stButton button {
    background-color: #00BFFF;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


st.title("214209T's Dengue Predictor - Sri Lanka Edition")
st.subheader("Custom ML App for Public Health Awareness")


st.sidebar.title("Dengue Prevention Tips")
st.sidebar.markdown("""
- Remove standing water from containers  
- Use mosquito nets during monsoon (Oct–Dec)  
- Report symptoms early  
- Source: Ministry of Health, Sri Lanka  
""")


districts = [
    'Ampara','Anuradhapura','Badulla','Batticaloa','Colombo',
    'Galle','Gampaha','Hambantota','Jaffna','Kalutara','Kandy',
    'Kegalle','Kilinochchi','Kurunegala','Mannar','Matale',
    'Matara','Moneragala','Mullaitivu','Nuwara Eliya',
    'Polonnaruwa','Puttalam','Ratnapura','Trincomalee','Vavuniya'
]

col1, col2 = st.columns(2)

with col1:
    district = st.selectbox('District', districts)
    year = st.number_input('Year', min_value=2019, max_value=2025, value=2023)
    month = st.slider('Month', 1, 12, value=6)
    latitude = st.number_input(
        "Latitude",
        min_value=5.0,
        max_value=10.0,
        value=6.92
    )
    longitude = st.number_input(
        "Longitude",
        min_value=79.0,
        max_value=82.0,
        value=79.91
    )

with col2:
    elevation = st.number_input('Elevation (m)', min_value=0, max_value=2000, value=4)
    temp_avg = st.number_input('Avg Temperature (°C)', min_value=20.0, max_value=35.0, value=28.0)
    precip_avg = st.number_input('Avg Precipitation (mm)', min_value=0.0, max_value=500.0, value=100.0)
    humidity_avg = st.number_input('Avg Humidity (%)', min_value=50.0, max_value=100.0, value=75.0)
    lagged_cases = st.number_input('Previous Month Cases', min_value=0, max_value=10000, value=100)

#PREDICTION 
if st.button('Predict Dengue Risk'):

    input_df = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'District': [district],
        'Latitude': [latitude],
        'Longitude': [longitude],
        'Elevation': [elevation],
        'Temp_avg': [temp_avg],
        'Precipitation_avg': [precip_avg],
        'Humidity_avg': [humidity_avg],
        'Lagged_Cases': [lagged_cases]
    })

    # Encode district
    district_encoded = encoder.transform(input_df[['District']])
    district_df = pd.DataFrame(district_encoded, columns=district_cols)

    input_encoded = pd.concat(
        [input_df.drop('District', axis=1).reset_index(drop=True),
         district_df.reset_index(drop=True)],
        axis=1
    )

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    input_dmatrix = xgb.DMatrix(input_scaled)
    prediction = model.predict(input_dmatrix)[0]

    st.success(f"Predicted Dengue Cases: {prediction:.0f}")

               # SHAP 

    # DataFrames
    input_scaled_df = pd.DataFrame(input_scaled, columns=features)
    input_display_df = input_encoded.copy()

    # TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled_df)

    # Convert to Series
    shap_series = pd.Series(shap_values[0], index=features)

   
    shap_series_clean = shap_series[~shap_series.index.str.startswith("District_")]
    display_values_clean = input_display_df.iloc[0][~input_display_df.columns.str.startswith("District_")]

   
    shap_series_clean["District"] = shap_series[
        shap_series.index.str.startswith("District_")
    ].sum()

    display_values_clean["District"] = district  # selected district name

    
    shap_explanation = shap.Explanation(
        values=shap_series_clean.values,
        base_values=explainer.expected_value,
        data=display_values_clean.values,
        feature_names=shap_series_clean.index.tolist()
    )

    fig, ax = plt.subplots(figsize=(8,5))
    shap.plots.waterfall(shap_explanation, show=False)
    st.pyplot(fig)