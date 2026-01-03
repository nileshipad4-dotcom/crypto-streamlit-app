import requests
import pandas as pd
import streamlit as st
from datetime import datetime

BINANCE_BASE = "https://api.binance.com"

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Crypto Momentum Scanner (Binance)")
st.caption("5m / 1h / 4h Percentage Change | Top 300 USDT Pairs")

# -------------------------------------------------
# Get top 300 USDT pairs by 24h volume
# -------------------------------------------------
@st.cache_data(ttl=300)
def get_top_symbols():
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr"
    data = requests.get(url, timeout=10).json()

    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")]
    df["quoteVolume"] = df["quoteVolume"].astype(float)

    return (
        df.sort_values("quoteVolume", ascending=False)
          .head(300)["symbol"]
          .tolist()
    )

# -------------------------------------------------
# Get price & % changes
# -------------------------------------------------
def get_price_changes(symbol):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 241
    }

    data = requests.get(url, params=params, timeout=10).json()
    closes = [float(k[4]) for k in data]

    current = closes[-1]
    price_5m = closes[-6]
    price_1h = closes[-61]
    price_4h = closes[0]

    return (
        round(current, 6),
        round((current - price_5m) / price_5m * 100, 2),
        round((current - price_1h) / price_1h * 100, 2),
        round((current - price_4h) / price_4h * 100, 2),
    )

# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------
@st.cache_data(ttl=120)
def build_table():
    symbols = get_top_symbols()
    rows = []

    for sym in symbols:
        try:
            price, c5, c1, c4 = get_price_changes(sym)
            rows.append({
                "Symbol": sym,
                "Price": price,
                "5m %": c5,
                "1h %": c1,
                "4h %": c4,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df = df.sort_values("5m %", ascending=False)
    return df

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
df = build_table()

st.subheader("📈 Full Scanner")
st.write(f"Last updated: **{datetime.utcnow()} UTC**")
st.dataframe(df, use_container_width=True, height=600)

# -------------------------------------------------
# TOP & BOTTOM 5
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔼 Top 5 (5m Momentum)")
    st.dataframe(df.head(5), use_container_width=True)

with col2:
    st.subheader("🔽 Bottom 5 (5m Momentum)")
    st.dataframe(df.tail(5).sort_values("5m %"), use_container_width=True)
