import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="BTC 3-Minute Candles", layout="wide")
st.title("🕯️ Bitcoin 3-Minute Candlestick Chart")

@st.cache_data(ttl=60)
def load_data():
    return yf.download(
        tickers="BTC-USD",
        period="7d",        # yfinance limits small intervals
        interval="3m"
    )

with st.spinner("Loading BTC data..."):
    df = load_data()

# Build candlestick chart
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="BTC-USD"
        )
    ]
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
