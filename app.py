import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt


st.set_page_config(page_title="Dengue Risk Analyzer", layout="wide", page_icon="🦟")


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
/* White background for a clean medical look */
.stApp {
    background-color: #FFFFFF;
}

/* Red Sidebar for Warning/Alert feel */
section[data-testid="stSidebar"] {
    background-color: #D32F2F !important;
}

section[data-testid="stSidebar"] .css-17l243g, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] p {
    color: white !important;
}

/* Headers in Medical Red */
h1, h2, h3 {
    color: #D32F2F !important;
    font-family: 'Helvetica Neue', sans-serif;
}

/* Info boxes style */
div.stAlert {
    border-radius: 0px;
    border-left: 5px solid #D32F2F;
}

/* Button Styling: Solid High-Contrast Red */
div.stButton > button {
    background-color: #D32F2F;
    color: white !important;
    border-radius: 4px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

div.stButton > button:hover {
    background-color: #B71C1C;
    border: none;
}

/* Label colors for white background */
label {
    color: #333333 !important;
    font-weight: bold !important;
}

/* Shadow boxes for inputs */
[data-testid="stVerticalBlock"] > div {
    background-color: #f9f9f9;
    padding: 15px;
    border: 1px solid #eeeeee;
}
</style>
""", unsafe_allow_html=True)


st.title(" Dengue Case Forecasting System")
st.markdown("##### *Public Health Decision Support Tool for Sri Lanka*")

# SIDEBAR
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2865/2865800.png", width=100)
st.sidebar.title("Awareness Panel")
st.sidebar.info("""
**Prevention Measures:**
- Empty flower pots weekly.
- Clean gutters and drains.
- Wear long sleeves during dawn/dusk.
""")

districts = [
    'Ampara','Anuradhapura','Badulla','Batticaloa','Colombo',
    'Galle','Gampaha','Hambantota','Jaffna','Kalutara','Kandy',
    'Kegalle','Kilinochchi','Kurunegala','Mannar','Matale',
    'Matara','Moneragala','Mullaitivu','Nuwara Eliya',
    'Polonnaruwa','Puttalam','Ratnapura','Trincomalee','Vavuniya'
]

# INPUT SECTION
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Geolocation & Time")
    district = st.selectbox('Select District', districts)
    year = st.number_input('Assessment Year', 2019, 2025, 2023)
    month = st.select_slider('Assessment Month', options=list(range(1, 13)), value=6)
    
    # Nested columns for Lat/Lon
    subcol1, subcol2 = st.columns(2)
    lat = subcol1.number_input("Lat", 5.0, 10.0, 6.92)
    lon = subcol2.number_input("Lon", 79.0, 82.0, 79.91)

with col2:
    st.subheader(" Environmental Factors")
    elevation = st.number_input('Elevation (m)', 0, 2000, 4)
    temp = st.slider('Avg Temp (°C)', 20.0, 35.0, 28.0)
    precip = st.slider('Avg Precipitation (mm)', 0.0, 500.0, 100.0)
    humid = st.slider('Avg Humidity (%)', 50.0, 100.0, 75.0)
    lagged = st.number_input('Cases in Previous Month', 0, 10000, 100)

st.write("---")

# PREDICTION
if st.button('GENERATE RISK ASSESSMENT'):

    input_df = pd.DataFrame({
        'Year': [year], 'Month': [month], 'District': [district],
        'Latitude': [lat], 'Longitude': [lon], 'Elevation': [elevation],
        'Temp_avg': [temp], 'Precipitation_avg': [precip],
        'Humidity_avg': [humid], 'Lagged_Cases': [lagged]
    })

    # Encoding & Scaling
    district_encoded = encoder.transform(input_df[['District']])
    district_df = pd.DataFrame(district_encoded, columns=district_cols)
    input_encoded = pd.concat([input_df.drop('District', axis=1).reset_index(drop=True), district_df.reset_index(drop=True)], axis=1)
    input_scaled = scaler.transform(input_encoded)

    # Predict
    input_dmatrix = xgb.DMatrix(input_scaled)
    prediction = model.predict(input_dmatrix)[0]

    # Result Display
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.error(f"### Predicted Cases: {prediction:.0f}")
        st.write("This forecast is based on historical weather patterns and current regional data.")

    with res_col2:
        # SHAP Plot
        input_scaled_df = pd.DataFrame(input_scaled, columns=features)
        input_display_df = input_encoded.copy()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled_df)
        shap_series = pd.Series(shap_values[0], index=features)

        shap_series_clean = shap_series[~shap_series.index.str.startswith("District_")]
        display_values_clean = input_display_df.iloc[0][~input_display_df.columns.str.startswith("District_")]
        shap_series_clean["District"] = shap_series[shap_series.index.str.startswith("District_")].sum()
        display_values_clean["District"] = district

        shap_explanation = shap.Explanation(
            values=shap_series_clean.values,
            base_values=explainer.expected_value,
            data=display_values_clean.values,
            feature_names=shap_series_clean.index.tolist()
        )

        fig, ax = plt.subplots(figsize=(8,4))
        # SHAP waterfall with medical colors
        shap.plots.waterfall(shap_explanation, show=False)
        st.pyplot(fig)