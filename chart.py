import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🕯️ BTC 3-Minute Candlestick Chart (Coinbase)")

@st.cache_data(ttl=60)
def load_data():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {"granularity": 180}  # 3 minutes

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Format: [ time, low, high, open, close, volume ]
    df = pd.DataFrame(
        data,
        columns=["time", "Low", "High", "Open", "Close", "Volume"]
    )

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time")

    return df.astype(float, errors="ignore")

df = load_data()

# Hard fail if empty
if df.empty:
    st.error("No data returned from Coinbase.")
    st.stop()

st.write(f"Loaded {len(df)} candles")

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
