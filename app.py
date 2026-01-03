# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# -------------------
# CONFIG
# -------------------
EXPIRIES = ["27-02-2026"]
UNDERLYINGS = ["BTC", "ETH"]
REFRESH_SECONDS = 60
API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}

st.set_page_config(
    page_title="Crypto Option Chain Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# IST TIME
# -------------------
def get_ist_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M:%S")

# -------------------
# LIVE PRICE FROM DELTA
# -------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        sym = "BTCUSD" if symbol == "BTC" else "ETHUSD"
        ticker = next(x for x in r if x["symbol"] == sym)
        return float(ticker["mark_price"])
    except:
        return None

# -------------------
# FETCH OPTION DATA
# -------------------
@st.cache_data(ttl=60)
def fetch_tickers(underlying: str, expiry: str) -> pd.DataFrame:
    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry}"
    )
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.json_normalize(r.json().get("result", []))

def safe_to_numeric(val):
    try:
        return pd.to_numeric(val)
    except:
        return val

# -------------------
# FORMAT OPTION CHAIN
# -------------------
def format_option_chain(df_raw: pd.DataFrame) -> pd.DataFrame:

    needed_cols = [
        "strike_price", "contract_type", "mark_price", "oi_contracts",
        "volume", "greeks.gamma", "greeks.delta", "greeks.vega"
    ]
    for c in needed_cols:
        if c not in df_raw.columns:
            df_raw[c] = None

    df = df_raw[needed_cols].copy()

    df.rename(columns={
        "greeks.gamma": "gamma",
        "greeks.delta": "delta",
        "greeks.vega": "vega"
    }, inplace=True)

    for c in ["strike_price", "mark_price", "oi_contracts", "volume", "gamma", "delta", "vega"]:
        df[c] = df[c].apply(safe_to_numeric)

    calls = df[df["contract_type"] == "call_options"].copy()
    puts  = df[df["contract_type"] == "put_options"].copy()

    calls = calls.rename(columns={
        "mark_price": "call_mark",
        "oi_contracts": "call_oi",
        "volume": "call_volume",
        "gamma": "call_gamma",
        "delta": "call_delta",
        "vega": "call_vega"
    }).drop(columns="contract_type")

    puts = puts.rename(columns={
        "mark_price": "put_mark",
        "oi_contracts": "put_oi",
        "volume": "put_volume",
        "gamma": "put_gamma",
        "delta": "put_delta",
        "vega": "put_vega"
    }).drop(columns="contract_type")

    merged = pd.merge(calls, puts, on="strike_price", how="outer")
    return merged.sort_values("strike_price").reset_index(drop=True)

# -------------------
# MAX PAIN
# -------------------
def compute_max_pain(df):
    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    U = []
    for i in range(len(df)):
        Q = -sum(A[i:] * B[i:])
        R = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        S = -sum(M[:i] * L[:i])
        T = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        U.append(round((Q + R + S + T) / 10000, 8))

    df["max_pain"] = U
    return df

# -------------------
# UI
# -------------------
st.title("📈 Crypto Option Chain — BTC & ETH (Live + Max Pain)")

selected_underlying = st.sidebar.selectbox("Underlying", UNDERLYINGS)
selected_expiry = st.sidebar.selectbox("Expiry", EXPIRIES)
auto_refresh = st.sidebar.checkbox("Auto-refresh", True)
download_raw = st.sidebar.checkbox("Download CSV", True)

# 🔥 SHOW BOTH LIVE PRICES
price_btc = get_delta_price("BTC")
price_eth = get_delta_price("ETH")

st.sidebar.metric("BTC Price (Delta)", f"{price_btc:,.2f}" if price_btc else "Error")
st.sidebar.metric("ETH Price (Delta)", f"{price_eth:,.2f}" if price_eth else "Error")

if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
    except:
        pass

# -------------------
# FETCH + PROCESS
# -------------------
raw = fetch_tickers(selected_underlying, selected_expiry)
df = format_option_chain(raw)
df = compute_max_pain(df)

# Add timestamp
df["timestamp_IST"] = get_ist_time()

# Save live CSV (RAW values)
os.makedirs("live", exist_ok=True)
df.to_csv(f"live/{selected_underlying}_live.csv", index=False)

# -------------------
# DISPLAY (8 DECIMALS EXCEPT STRIKE)
# -------------------
df_display = df.copy()

for col in df_display.columns:
    if col != "strike_price" and pd.api.types.is_numeric_dtype(df_display[col]):
        df_display[col] = df_display[col].map(lambda x: f"{x:.8f}" if pd.notna(x) else "")

st.subheader(f"{selected_underlying} — Expiry {selected_expiry}")
st.dataframe(df_display, use_container_width=True)

if download_raw:
    st.download_button(
        "Download Live CSV",
        df.to_csv(index=False),
        file_name=f"{selected_underlying}_live.csv",
        mime="text/csv"
    )

st.caption("Live data auto-exported every refresh • Delta Exchange")








