import streamlit as st
import requests
import pandas as pd

st.title("DEBUG: Binance 3m data")

url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "BTCUSDT",
    "interval": "3m",
    "limit": 5
}

resp = requests.get(url, params=params, timeout=20)

st.write("Status code:", resp.status_code)
st.write("Raw response:")
st.json(resp.json())

