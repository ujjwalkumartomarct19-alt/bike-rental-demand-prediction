import streamlit as st
import pandas as pd
import joblib
import os
import base64

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Bike Rental Prediction", layout="wide")





# ================= BACKGROUND + CSS =================
def set_background(image_name):
    image_path = os.path.join(os.path.dirname(__file__), image_name)

    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as f:
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

        /* dark overlay */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.35);
            z-index: -1;
        }}

        /* rounded main container */
        .block-container {{
            border-radius: 40px;
            padding: 2rem;
        }}

        /* rounded prediction box */
        .rounded-box {{
            background: white;
            border-radius: 50px;
            padding: 30px;
            margin-top: 40px;
            box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# APPLY BACKGROUND
set_background("bike_bg.png")

# ================= LOAD MODEL & SCALER =================
@st.cache_resource
def load_files():
    base = os.path.dirname(__file__)

    model_path = os.path.join(base, "bike_model.joblib")
    scaler_path = os.path.join(base, "scaler.joblib")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


# ⚠️ DO NOT REMOVE THIS
model, scaler = load_files()

# ================= SIDEBAR INPUTS =================
st.sidebar.title("Input Parameters")

season = st.sidebar.selectbox("Season", [0, 1, 2, 3])
yr = st.sidebar.selectbox("Year", [0, 1])
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 8)
holiday = st.sidebar.selectbox("Holiday", [0, 1])
weekday = st.sidebar.slider("Weekday", 0, 6, 4)
workingday = st.sidebar.selectbox("Working Day", [0, 1])
weathersit = st.sidebar.selectbox("Weather", [0, 1, 2, 3])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

predict_btn = st.sidebar.button("Predict")

# ================= INPUT DATA =================
df = pd.DataFrame([{
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
}])

# ================= MAIN UI =================
st.markdown(
    "<h1 style='color:#000000; font-weight:800;'>🚲 Bike Rental Demand Prediction</h1>",
    unsafe_allow_html=True
)

st.dataframe(df, use_container_width=True)

# ================= PREDICTION (BOTTOM BOX) =================
if predict_btn:
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    st.markdown(
        f"""
        <div class="rounded-box">
            <h2 style="color:#000000;">✅ Predicted Bike Rentals</h2>
            <h1 style="color:#2E7D32; font-size:48px;">{int(prediction)}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )



