import streamlit as st
import pandas as pd
from binance.client import Client
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🕯️ BTC 3-Minute Candlestick Chart (Binance)")

client = Client()  # public endpoints only, no API key needed

@st.cache_data(ttl=60)
def load_data():
    klines = client.get_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_3MINUTE,
        limit=500
    )

    df = pd.DataFrame(klines, columns=[
        "time", "Open", "High", "Low", "Close", "Volume",
        "_", "_", "_", "_", "_", "_"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["Open", "High", "Low", "Close", "Volume"]] = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)

    return df

with st.spinner("Loading Binance data..."):
    df = load_data()

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
    yaxis_title="Price (USDT)",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
