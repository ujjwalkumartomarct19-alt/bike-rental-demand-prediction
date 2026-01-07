import streamlit as st
import pandas as pd
import joblib
import os
import base64


# ================= VALUE MEANINGS =================

holiday_map = {
    "No Holiday": 0,
    "Holiday": 1
}

workingday_map = {
    "No (Weekend / Holiday)": 0,
    "Yes (Working Day)": 1
}

weekday_map = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}

weather_map = {
    "Clear / Few Clouds": 0,
    "Mist / Cloudy": 1,
    "Light Rain / Snow": 2,
    "Heavy Rain / Snow": 3
}


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
# ================= SIDEBAR INPUTS =================
st.sidebar.title("Input Parameters")

# ----- readable mappings -----
season_map = {
    "Spring": 0,
    "Summer": 1,
    "Fall": 2,
    "Winter": 3
}

year_map = {
    "2011": 0,
    "2012": 1
}

holiday_map = {
    "No Holiday": 0,
    "Holiday": 1
}

workingday_map = {
    "No (Weekend / Holiday)": 0,
    "Yes (Working Day)": 1
}

weekday_map = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}

weather_map = {
    "Clear / Few Clouds": 0,
    "Mist / Cloudy": 1,
    "Light Rain / Snow": 2,
    "Heavy Rain / Snow": 3
}

# ----- sidebar inputs (user-friendly) -----
season_label = st.sidebar.selectbox("Season", list(season_map.keys()))
season = season_map[season_label]

yr_label = st.sidebar.selectbox("Year", list(year_map.keys()))
yr = year_map[yr_label]

mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 8)

holiday_label = st.sidebar.selectbox("Holiday", list(holiday_map.keys()))
holiday = holiday_map[holiday_label]

weekday_label = st.sidebar.selectbox("Weekday", list(weekday_map.keys()))
weekday = weekday_map[weekday_label]

workingday_label = st.sidebar.selectbox("Working Day", list(workingday_map.keys()))
workingday = workingday_map[workingday_label]

weather_label = st.sidebar.selectbox("Weather Condition", list(weather_map.keys()))
weathersit = weather_map[weather_label]

temp = st.sidebar.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
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




