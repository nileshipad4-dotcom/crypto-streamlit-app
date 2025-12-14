# app.py
import streamlit as st
import pandas as pd
import requests
from typing import List
import io
import os

# -------------------
# CONFIG
# -------------------
EXPIRIES = ["11-12-2025"]
UNDERLYINGS = ["BTC", "ETH"]
REFRESH_SECONDS = 30
API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}

st.set_page_config(
    page_title="Crypto Option Chain Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# DELTA EXCHANGE PRICE
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
def fetch_tickers(underlying, expiry):
    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry}"
    )
    r = requests.get(url, headers=HEADERS, timeout=20)
    return pd.json_normalize(r.json().get("result", []))

def safe_num(x):
    try:
        return pd.to_numeric(x)
    except:
        return 0

def format_chain(df):
    df = df[[
        "strike_price","contract_type","mark_price",
        "oi_contracts","greeks.gamma","greeks.delta","greeks.vega"
    ]]
    df = df.applymap(safe_num)

    calls = df[df.contract_type == "call_options"]
    puts  = df[df.contract_type == "put_options"]

    calls = calls.rename(columns={
        "mark_price":"call_mark","oi_contracts":"call_oi"
    }).drop(columns="contract_type")

    puts = puts.rename(columns={
        "mark_price":"put_mark","oi_contracts":"put_oi"
    }).drop(columns="contract_type")

    return pd.merge(calls, puts, on="strike_price").sort_values("strike_price")

# -------------------
# MAX PAIN
# -------------------
def compute_max_pain(df):
    A = df.call_mark.values
    B = df.call_oi.values
    G = df.strike_price.values
    L = df.put_oi.values
    M = df.put_mark.values

    U = []
    for i in range(len(df)):
        Q = -sum(A[i:] * B[i:])
        R = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        S = -sum(M[:i] * L[:i])
        T = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        U.append(round((Q + R + S + T) / 10000))

    df["max_pain"] = U
    return df

# -------------------
# UI
# -------------------
st.title("📈 Live Option Chain + Max Pain")

underlying = st.sidebar.selectbox("Underlying", UNDERLYINGS)
expiry = st.sidebar.selectbox("Expiry", EXPIRIES)

price = get_delta_price(underlying)
st.sidebar.metric(f"{underlying} Price (Delta)", f"{price:,.2f}" if price else "Error")

df_raw = fetch_tickers(underlying, expiry)
df = compute_max_pain(format_chain(df_raw))

# -------------------
# 🔥 SHARE LIVE DATA
# -------------------
os.makedirs("shared", exist_ok=True)
df[["strike_price","max_pain"]].to_csv(
    "shared/live_max_pain.csv",
    index=False
)

st.success("Live Max Pain written to shared/live_max_pain.csv")

st.dataframe(df, use_container_width=True)

st.caption("Auto-updates every 30 seconds")

