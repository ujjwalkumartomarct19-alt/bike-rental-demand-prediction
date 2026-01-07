import streamlit as st
import pandas as pd
import joblib
import os
import base64

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Bike Rental Demand Prediction", layout="wide")

# ================= BACKGROUND + GLOBAL CSS =================
def set_background(image_name):
    image_path = os.path.join(os.path.dirname(__file__), image_name)

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        body {{
            background-color: #0E1117;
        }}

        .stApp {{
            background-color: #0E1117;
        }}

        /* MAIN CARD */
        .main-card {{
            background: #111827;
            border-radius: 40px;
            padding: 25px;
            margin-top: 20px;
        }}

        /* IMAGE BANNER */
        .banner {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            border-radius: 30px;
            padding: 35px;
            height: 330px;
            display: flex;
            align-items: flex-start;
        }}

        .banner-title {{
            font-size: 34px;
            font-weight: 800;
            color: #111111;
            background: rgba(255,255,255,0.75);
            padding: 12px 22px;
            border-radius: 18px;
        }}

        /* PREDICTION CARD */
        .pred-card {{
            background: white;
            border-radius: 28px;
            padding: 30px;
            margin-top: 25px;
            box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
            text-align: center;
        }}

        .pred-title {{
            font-size: 22px;
            font-weight: 700;
            color: #111;
        }}

        .pred-value {{
            font-size: 52px;
            font-weight: 900;
            color: #2E7D32;
            margin-top: 10px;
        }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            background-color: #111827;
            border-radius: 25px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# APPLY BACKGROUND IMAGE (YOUR BANNER IMAGE)
set_background("bike_bg.png")

# ================= LOAD MODEL =================
@st.cache_resource
def load_files():
    base = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base, "bike_model.joblib"))
    scaler = joblib.load(os.path.join(base, "scaler.joblib"))
    return model, scaler

model, scaler = load_files()

# ================= SIDEBAR INPUTS =================
st.sidebar.title("Input Parameters")

holiday = st.sidebar.selectbox("Holiday", [0, 1])
weekday = st.sidebar.selectbox("Weekday", [0, 1, 2, 3, 4, 5, 6])
workingday = st.sidebar.selectbox("Working Day", [0, 1])
weathersit = st.sidebar.selectbox("Weather", [0, 1, 2, 3])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.33)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.68)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.20)

predict_btn = st.sidebar.button("Predict")

# ================= INPUT DATA =================
df = pd.DataFrame([{
    "season": 1,
    "yr": 0,
    "mnth": 6,
    "hr": 8,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "hum": hum,
    "windspeed": windspeed
}])

# ================= MAIN UI =================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# IMAGE BANNER
st.markdown(
    """
    <div class="banner">
        <div class="banner-title">🚲 Bike Rental Demand Prediction</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= PREDICTION =================
if predict_btn:
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-title">✅ Predicted Bike Rentals</div>
            <div class="pred-value">{int(prediction)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

