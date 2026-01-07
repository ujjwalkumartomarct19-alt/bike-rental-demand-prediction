importimport streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import base64

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bike Rental Prediction",
    layout="wide"
)

# ================= BACKGROUND IMAGE =================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* watermark effect */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.75);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# apply background
set_bg("bike_bg.png")

# ================= LOAD MODEL =================
@st.cache_resource
def load_files():
    model = pickle.load(open("bike_rental_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_files()

# ================= SIDEBAR =================
st.sidebar.title("Input Parameters")

season = st.sidebar.selectbox("Season", [0,1,2,3])
yr = st.sidebar.selectbox("Year", [0,1])
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 8)
holiday = st.sidebar.selectbox("Holiday", [0,1])
weekday = st.sidebar.slider("Weekday", 0, 6, 4)
workingday = st.sidebar.selectbox("Working Day", [0,1])
weathersit = st.sidebar.selectbox("Weather Situation", [0,1,2,3])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

predict_btn = st.sidebar.button("Predict")

# ================= INPUT DATA =================
input_data = {
    "season": season,
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "hum": hum,
    "windspeed": windspeed
}

df = pd.DataFrame([input_data])

# ================= MAIN UI =================
st.markdown("## 🚲 Bike Rental Demand Prediction")

tab1, tab2 = st.tabs(["Info", "Prediction"])

# -------- INFO TAB --------
with tab1:
    st.markdown("""
    **This application predicts bike rental demand based on weather and time-related inputs.**

    - Seasonality and weather conditions strongly affect bike usage  
    - Working days and hours capture commuting behavior  
    - Model is trained using historical bike rental data  
    """)

# -------- PREDICTION TAB --------
with tab2:
    st.subheader("Input Summary")
    st.dataframe(df, use_container_width=True)

    if predict_btn:
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)[0]

        st.success(f"🚴 Predicted Bike Rentals: **{int(prediction)}**")


