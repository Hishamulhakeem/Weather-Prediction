import streamlit as st
import pickle
import datetime
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Weather Prediction App",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        color: white;
        padding-bottom: 20px;
    }
    .centered-text {
        text-align: center;
    }
    .prediction-container {
        margin-top: 20px;
        padding: 20px;
        border-radius: 8px;
        background-color: #1E1E1E;
        color: white;
    }
    .weather-metric {
        padding: 10px;
        margin: 8px 0;
        border-radius: 6px;
        background-color: #2D2D2D;
        color: white;
    }
    div[data-testid="stForm"] {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
    }
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        margin: 0 auto;
        display: block;
    }
    div.stSelectbox > div[data-baseweb="select"] > div {
        background-color: #2D2D2D;
        color: white;
    }
    div[data-testid="stDateInput"] > div > div > input {
        background-color: #2D2D2D;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_model(location):
    try:
        model_path = f"Classifier{location.replace(' ', '')}.pkl"
        
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except:
                with open(model_path, "rb") as file:
                    return pickle.load(file)
        else:
            st.error(f"Model file not found: {model_path}")
            class DummyModel:
                def predict(self, features):
                    if location == "Austin":
                        return np.array([[19.83, 29.42, 9.78, 60.80, 3.09, 4]])
                    else:
                        return np.array([[20.68, 32.52, 0.00, 70.5, 9.6]])
            return DummyModel()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
    
def format_prediction(prediction, location):
    if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
        prediction = prediction.flatten()
    
    event_mapping = {
        0: "Clear",
        1: "Cloudy", 
        2: "Rain",
        3: "Thunderstorm",
        4: "Fog",
        5: "Snow",
        6: "Drizzle"
    }
    
    results = {}

    if location == "Austin":
        if len(prediction) >= 6:
            avg_temp = prediction[1]
            max_temp = prediction[0]
            min_temp = prediction[2]
            humidity = prediction[3]
            wind_speed = prediction[4]
            event_code = int(prediction[5])            
        event = event_mapping.get(event_code, "Fog")
        
        results = {
            "avg_temp": f"{avg_temp:.2f}°C",
            "max_temp": f"{max_temp:.2f}°C",
            "min_temp": f"{min_temp:.2f}°C",
            "humidity": f"{humidity:.2f}%",
            "wind": f"{wind_speed:.2f} MPH",
            "event": event
        }
    else:  
         if len(prediction) >= 5:
            rain = prediction[0]
            max_temp = prediction[1]
            min_temp = prediction[2]
            humidity = prediction[3]
            wind_speed = prediction[4]
            
            results = {
                "rain": f"{rain:.2f}%",
                "max_temp": f"{max_temp:.2f}°C",
                "min_temp": f"{min_temp:.2f}°C",
                "humidity": f"{humidity:.2f}%",
                "wind": f"{wind_speed:.2f} km/h"
            }
    
    return results

st.markdown('<h1 class="main-header">Weather Prediction App</h1>', unsafe_allow_html=True)

with st.form(key='prediction_form'):
    st.markdown('<p class="centered-text">Select city and date for weather prediction</p>', unsafe_allow_html=True)

    location = st.selectbox(
        "City",
        ["Bengaluru", "Austin"],
        index=0
    )
    
    date = st.date_input(
        "Date",
        datetime.date.today(),
        min_value=datetime.date.today(),
        max_value=datetime.date.today() + datetime.timedelta(days=7)
    )
    
    submit_button = st.form_submit_button(label="Predict Weather")

prediction_made = False
formatted_results = {}

if submit_button:
    with st.spinner("Predicting weather..."):
        model = load_model(location)
        
        if model is not None:
            features = np.array([[date.year, date.month, date.day]])
            try:
                prediction = model.predict(features)
                formatted_results = format_prediction(prediction, location)
                prediction_made = True
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                with st.expander("Debug Information"):
                    st.write(f"Model type: {type(model)}")
                    st.write(f"Features shape: {features.shape}")
                    st.write(f"Prediction: {prediction if 'prediction' in locals() else 'Not available'}")
if prediction_made or 'formatted_results' in locals():
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown(f"<h3 class='centered-text'>Weather Forecast</h3>", unsafe_allow_html=True)
    st.markdown(f"<p class='centered-text'><b>{location}</b> • {date.strftime('%A, %B %d, %Y')}</p>", unsafe_allow_html=True)
    if location == "Austin" and formatted_results:
        st.markdown(f"<div class='weather-metric'>Predicted Max Temperature: <b>{formatted_results.get('max_temp', '29.42°C')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Avg Temperature: <b>{formatted_results.get('avg_temp', '19.83°C')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Low Temperature: <b>{formatted_results.get('min_temp', '9.78°C')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Humidity: <b>{formatted_results.get('humidity', '60.80%')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted wind: <b>{formatted_results.get('wind', '3.09 MPH')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Event : <b>{formatted_results.get('event', 'Fog')}</b></div>", unsafe_allow_html=True)
    elif location == "Bengaluru" and formatted_results:
        st.markdown(f"<div class='weather-metric'>Predicted Rainfall: <b>{formatted_results.get('rain', '0.00%')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Max Temperature: <b>{formatted_results.get('max_temp', '32.52°C')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Min Temperature: <b>{formatted_results.get('min_temp', '20.68°C')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Humidity: <b>{formatted_results.get('humidity', '70.50%')}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-metric'>Predicted Wind Speed: <b>{formatted_results.get('wind', '9.60 km/h')}</b></div>", unsafe_allow_html=True)

    
    
    st.markdown('</div>', unsafe_allow_html=True)
