import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🕯️ BTC 3-Minute Candlestick Chart")

@st.cache_data(ttl=60)
def load_data():
    df = yf.download(
        "BTC-USD",
        interval="3m",
        period="5d",   # keep this small
        progress=False
    )
    df = df.dropna()  # CRITICAL
    return df

with st.spinner("Fetching data..."):
    df = load_data()

# 🚨 HARD STOP if no data
if df.empty:
    st.error("No 3-minute data returned. Refresh in 1–2 minutes.")
    st.stop()

st.write(f"Loaded {len(df)} candles")

fig = go.Figure(
    go.Candlestick(
        x=df.index,
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
