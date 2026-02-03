import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="BTC 5-Minute Candles", layout="wide")
st.title("🕯️ Bitcoin 5-Minute Candlestick Chart (Coinbase)")

@st.cache_data(ttl=60)
def load_data():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {
        "granularity": 300,  # ✅ 5 minutes (SUPPORTED)
        "limit": 300
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    r = requests.get(url, params=params, headers=headers, timeout=20)

    if r.status_code != 200:
        st.error(f"Coinbase API error: {r.status_code}")
        st.write(r.text)
        st.stop()

    data = r.json()

    # Format: [ time, low, high, open, close, volume ]
    df = pd.DataFrame(
        data,
        columns=["time", "Low", "High", "Open", "Close", "Volume"]
    )

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time")

    df[["Open", "High", "Low", "Close", "Volume"]] = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)

    return df

df = load_data()

if df.empty:
    st.error("No candle data returned.")
    st.stop()

st.write(f"Loaded {len(df)} candles")

fig = go.Figure(
    go.Candlestick(
        x=df["time"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="green",
        decreasing_line_color="red"
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
