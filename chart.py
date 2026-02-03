import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🕯️ BTC 3-Minute Candlestick Chart (Binance REST)")

@st.cache_data(ttl=60)
def load_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "3m",
        "limit": 500
    }

    data = requests.get(url, params=params, timeout=20).json()

    df = pd.DataFrame(data, columns=[
        "time", "Open", "High", "Low", "Close", "Volume",
        "_", "_", "_", "_", "_", "_"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["Open", "High", "Low", "Close", "Volume"]] = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)

    return df

df = load_data()

fig = go.Figure(
    go.Candlestick(
        x=df["time"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )
)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
