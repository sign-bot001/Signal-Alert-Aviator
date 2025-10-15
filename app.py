# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Signal Alert AVTR", layout="wide")
tz_choice = "Asia/Seoul"

# ----------------- CSS -----------------
st.markdown("""
<style>
body {background-color: #050608;}
.stApp { color: #cfeef8; }
.title {font-family: 'Courier New', monospace; color: #7afcff; font-size:28px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚡ Signal Alert AVTR — Version Optimisée</div>', unsafe_allow_html=True)
st.write("Clique sur le bouton pour entraîner le modèle et lancer les prédictions.")

# ----------------- SAMPLE HISTORY -----------------
SAMPLE_HISTORY = [1.3,1.23,1.56,2.25,1.15,13.09,20.91,2.05,10.17,3.82,
                  1,1.46,1.4,1.73,1.17,1.00,26.60,8.6,1.27,1.46,
                  1.36,1.76,3.61,2.74,1.47,3.7,1.05]

@st.cache_data
def load_sample():
    tz = pytz.timezone(tz_choice)
    last_ts = datetime.now(tz)
    timestamps = [last_ts - timedelta(minutes=(len(SAMPLE_HISTORY)-1-i)) for i in range(len(SAMPLE_HISTORY))]
    df = pd.DataFrame({"timestamp":timestamps, "multiplier":SAMPLE_HISTORY})
    return df

# ----------------- FEATURE ENGINEERING -----------------
def make_features(series, lags=10):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# ----------------- MODEL -----------------
@st.cache_resource
def train_model(series, lags=10, n_estimators=150):
    X, y = make_features(series, lags)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, scaler, mae, rmse

def predict_future(model, scaler, series, lags=10, steps=10):
    preds = []
    window = series[-lags:].tolist()
    for _ in range(steps):
        X = scaler.transform([window])
        p = model.predict(X)[0]
        preds.append(p)
        window.append(p)
        window = window[-lags:]
    return preds

# ----------------- UI -----------------
df = load_sample()
st.dataframe(df.tail())

go = st.button("⚙️ Entraîner et prédire")

if go:
    st.info("Entraînement du modèle en cours...")
    model, scaler, mae, rmse = train_model(df["multiplier"].values, lags=10)
    st.success(f"Modèle entraîné (MAE={mae:.3f}, RMSE={rmse:.3f})")

    preds = predict_future(model, scaler, df["multiplier"].values, lags=10, steps=20)
    tz = pytz.timezone(tz_choice)
    future_times = [df["timestamp"].iloc[-1] + timedelta(minutes=i+1) for i in range(len(preds))]
    out = pd.DataFrame({"timestamp_kst":future_times, "predicted_multiplier":preds})
    st.dataframe(out)

    st.line_chart(out.set_index("timestamp_kst"))
    st.download_button("Télécharger CSV des prédictions",
                       out.to_csv(index=False),
                       file_name="predictions_signal_alert_avtr.csv")
