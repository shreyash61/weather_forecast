import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
import pytz
import requests
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


# Function to convert image to base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Set the image path (same folder as this script)
image_path = "bg.jpg"
b64_string = get_base64(image_path)
    
# ================= Streamlit Configuration ===================
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/jpg;base64,{b64_string}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }
    .main {
        font-size: 20px;
        color: #ffffff;
        background-color: blue;
    }
    .stPlot {
            width: 80%;
            height: 400px;
        }
    h1, h2, h3 {
        color: #f7f7f7;
        text-shadow: 1px 1px 2px #333;
    }
    
    
    
    
    </style>
""", unsafe_allow_html=True)


# set_background('bg.jpg')
# print("background loaded")

# ================ Weather API Configuration ===================
API_key = "11f0f3da9ff1dd09178dd2857c253f5b"
BASE_URL = "https://api.openweathermap.org/data/2.5/"

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    current_temp = data["main"]["temp"]
    feels_min = data["main"]["feels_like"]
    temp_min = data["main"]["temp_min"]
    temp_max = data["main"]["temp_max"]
    return {
        "city": data["name"],
        "current_temp": round(current_temp),
        "feels_min": round(feels_min),
        "temp_min": round(temp_min),
        "temp_max": round(temp_max),
        "humidity": round(data["main"]["humidity"]),
        "description": data["weather"][0]["description"],
        "country": data["sys"]["country"],
        "wind_gust_dir": data["wind"]["deg"],
        "pressure": data["main"]["pressure"],
        "wind_gust_speed": data["wind"]["speed"]
    }

# =============== Load and Prepare Historical Data ================
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data["WindGustDir"] = le.fit_transform(data["WindGustDir"])
    data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])
    X = data[["MinTemp", "MaxTemp", "WindGustDir", "WindGustSpeed", "Humidity", "Pressure", "Temp"]]
    y = data["RainTomorrow"]
    return X, y, le

def train_rainmodel(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

# ================= Streamlit App ====================

def main():
    st.image("logo.png", width=100)
    st.title("🌤️ Weather Forecast Dashboard")
    city = st.text_input("Enter City Name", "London")

    if st.button("Get Forecast"):
        current_weather = get_current_weather(city)
        historical_data = read_historical_data("data.csv")

        # Prepare and train rain model
        X, y, le = prepare_data(historical_data)
        rain_model = train_rainmodel(X, y)

        # Map wind direction
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25), ("ENE", 56.25, 78.75),
            ("E", 78.75, 101.25), ("ESE", 101.25, 123.75), ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75),
            ("S", 168.75, 191.25), ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25), ("NNW", 326.25, 348.75)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
            "MinTemp": current_weather["temp_min"],
            "MaxTemp": current_weather["temp_max"],
            "WindGustDir": compass_direction_encoded,
            "WindGustSpeed": current_weather["wind_gust_speed"],
            "Humidity": current_weather["humidity"],
            "Pressure": current_weather["pressure"],
            "Temp": current_weather["current_temp"]
        }

        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]

        # Train regression models for future predictions
        X_temp, y_temp = prepare_regression_data(historical_data, "Temp")
        X_hum, y_hum = prepare_regression_data(historical_data, "Humidity")
        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)
        future_temp = predict_future(temp_model, current_weather["temp_min"])
        future_humidity = predict_future(hum_model, current_weather["humidity"])

        # Prepare time
        timeZone = pytz.timezone("Asia/Karachi")
        now = datetime.now(timeZone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # Display weather results
        st.subheader(f"📍 Weather in {city}, {current_weather['country']}")
        st.markdown(f"""
        <div style='font-size:22px;'>
        🌡️ Current Temperature: {current_weather['current_temp']}°C<br>
        🤗 Feels Like: {current_weather['feels_min']}°C<br>
        📉 Min Temp: {current_weather['temp_min']}°C<br>
        📈 Max Temp: {current_weather['temp_max']}°C<br>
        💧 Humidity: {current_weather['humidity']}%<br>
        🌥️ Weather: {current_weather['description']}<br>
        🌧️ Rain Prediction: <b>{'Yes' if rain_prediction else 'No'}</b><br>
        </div>
        """, unsafe_allow_html=True)

        # Display rain image
        rain_image = "rain.jpeg" if rain_prediction else "clear.jpg"
        st.image(rain_image,  width=500)

        # Plot temperature
        col1, col2 = st.columns(2)
        with col1:
         st.subheader("🌡️ Next 5 Hour Forecast - Temperature")
         fig1, ax1 = plt.subplots(figsize=(6, 4))
         ax1.plot(future_times, future_temp, marker='o', color='orange')
         ax1.set_title("Temperature Forecast", fontsize=14)
         ax1.set_xlabel("Hour")
         ax1.set_ylabel("Temp (°C)")
         ax1.grid(True)
         st.pyplot(fig1)

        # Plot humidity
        with col2:
         st.subheader("💧 Next 5 Hour Forecast - Humidity")
         fig2, ax2 = plt.subplots(figsize=(6, 4))
         ax2.plot(future_times, future_humidity, marker='s', color='blue')
         ax2.set_title("Humidity Forecast", fontsize=14)
         ax2.set_xlabel("Hour")
         ax2.set_ylabel("Humidity (%)")
         ax2.grid(True)
         st.pyplot(fig2)

if __name__ == '__main__':
    main()
