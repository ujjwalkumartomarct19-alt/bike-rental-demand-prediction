import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page Config

st.set_page_config(
    page_title="Bike Rental Regression",
    layout="wide"
)

@st.cache_resource
def load_files():
    model = pickle.load(open("bike_model_optimized.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_files()

# Sidebar Inputs

st.sidebar.title("Input Parameters")

season = st.sidebar.selectbox("Season", [0,1,2,3])
yr = st.sidebar.selectbox("Year",[2011,2012])
mnth = st.sidebar.slider("Month",1,12,6)
hr = st.sidebar.slider("Hour",0,23,8)
holiday = st.sidebar.selectbox("Holiday",[0,1])
weekday = st.sidebar.slider("Weekday",0,6,4)
workingday = st.sidebar.selectbox("Workingday",[0,1])
weathersit = st.sidebar.selectbox("Weathersit", [0,1,2,3])
temp = st.sidebar.slider("Temperature",0.0,1.0,0.5)
hum = st.sidebar.slider("Humidity",0.0,1.0,0.5)
windspeed = st.sidebar.slider("Windspeed",0.0,1.0,0.5)

predict_btn = st.sidebar.button("Predict")

# Input DataFrame

input_data = {
    "season" : season,        
    "yr" : yr,         
    "mnth" : mnth,      
    "hr" : hr,   
    "holiday" : holiday,        
    "weekday" : weekday,
    "workingday" : workingday,
    "weathersit" : weathersit,
    "temp" : temp,
    "hum" : hum,  
    "windspeed" : windspeed
}
    
df = pd.DataFrame([input_data])

# Main UI

st.title("Bike Rentals Prediction System")

tab1, tab2= st.tabs(["Info","Prediction"])

# -------- Info Section --------

with tab1:
    st.subheader("Feature Information")

    st.markdown("""
    
        **Season** 
        - Indicates the season of the year.
        - fall = 0, springer = 1, summer = 2, winter = 3
                
        **Year** 
        - Year indicator.
                
        **Month** 
        - Represents monthly trends that influence bike rental demand.
        - 1 to 12 as jan to Dec.
                
        **Hour** 
        - Hour of the day capturing commuting patterns.
        - using 0 to 23 hour pattern.
                
        **Holiday** 
        - Indicates whether the day is a holiday or not.
        - 0 = No, 1 = Yes.
                
        **Weekday** 
        - Day of the week.
        - 0 to 6 as Sunday to Saturday.
                
        **Workingday** 
        - Indicates regular working days or not.
        - 0 = No, 1 = Yes. 
                
        **Weathersit** 
        - Weather condition of the day.
        - Clear = 0, Heavy Rain = 1, Light Snow = 2, Mist = 3.
                
        **Temperature** 
        - Temperature of the day.
                
        **Humidity** 
        - Humidity level of the day.
                
        **Windspeed** 
        - Wind condition of the day.
    """)

# -------- Prediction --------

with tab2:
    st.subheader("Prediction")

    st.dataframe(df.T, use_container_width=True)

    if predict_btn:
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)[0]
        

        st.success(f" Predicted Bike Rentals: **{round(prediction)}**")
