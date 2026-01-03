import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(layout="wide")
st.title("📊 Crypto % Change – Last 5 Minutes (Binance)")

# Auto refresh every 60 seconds
st_autorefresh(interval=60_000, key="refresh")

BASE_URL = "https://api.binance.com"

# ==================================================
# HELPERS
# ==================================================
@st.cache_data(ttl=300)
def get_all_symbols():
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    data = requests.get(url, timeout=10).json()
    return [
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]

def get_5m_change(symbol):
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 6
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if len(data) < 6:
        return None

    price_5m_ago = float(data[0][4])
    current_price = float(data[-1][4])

    return round(((current_price - price_5m_ago) / price_5m_ago) * 100, 3)

# ==================================================
# MAIN LOGIC
# ==================================================
symbols = get_all_symbols()

results = []
progress = st.progress(0)

for i, sym in enumerate(symbols):
    try:
        change = get_5m_change(sym)
        if change is not None:
            results.append({
                "Symbol": sym,
                "Change_5m_%": change
            })
    except Exception:
        pass

    progress.progress((i + 1) / len(symbols))

df = pd.DataFrame(results)

if df.empty:
    st.warning("No data received. Try refreshing.")
    st.stop()

df = df.sort_values("Change_5m_%", ascending=False).reset_index(drop=True)

# ==================================================
# CSV EXPORT
# ==================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download CSV",
    data=csv_data,
    file_name=f"crypto_5m_change_{timestamp}.csv",
    mime="text/csv"
)

# ==================================================
# DISPLAY
# ==================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Top Gainers (5m)")
    st.dataframe(
        df.head(20),
        use_container_width=True,
        height=600
    )

with col2:
    st.subheader("🔻 Top Losers (5m)")
    st.dataframe(
        df.tail(20).sort_values("Change_5m_%"),
        use_container_width=True,
        height=600
    )
