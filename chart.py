import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.subheader("📈 BTC Live OHLC Chart")

# -----------------------------
# TIMEFRAME DROPDOWN
# -----------------------------
tf_map = {
    "3 Min": "3m",
    "5 Min": "5m",
    "10 Min": "10m",
    "15 Min": "15m"
}

tf_label = st.selectbox("Select Timeframe", list(tf_map.keys()))
interval = tf_map[tf_label]

# -----------------------------
# FETCH OHLC DATA
# -----------------------------
@st.cache_data(ttl=30)
def fetch_ohlc(interval):
    url = "https://api.india.delta.exchange/v2/history/candles"
    params = {
        "symbol": "BTCUSD",
        "resolution": interval,
        "limit": 200
    }
    r = requests.get(url, params=params, timeout=10).json()
    data = r.get("result", [])

    df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

df = fetch_ohlc(interval)

# -----------------------------
# BUILD CANDLE CHART
# -----------------------------
fig = go.Figure(data=[
    go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )
])

fig.update_layout(
    title=f"BTCUSD {tf_label} OHLC",
    xaxis_title="Time",
    yaxis_title="Price",
    height=600,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
