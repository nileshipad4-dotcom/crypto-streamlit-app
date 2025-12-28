# crypto compare

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

# -------------------------------------------------
# AUTO REFRESH (60s)
# -------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60 * 1000, key="refresh")
except ImportError:
    pass

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

def rotated_time_sort(times, pivot="17:30"):
    pivot_minutes = int(pivot[:2]) * 60 + int(pivot[3:])
    def key(t):
        h, m = map(int, t.split(":"))
        return ((h * 60 + m) - pivot_minutes) % (24 * 60)
    return sorted(times, key=key, reverse=True)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "28-12-2025"
ASSETS = ["BTC", "ETH"]

STRIKE_COL_IDX = 6
TIMESTAMP_COL_IDX = 14
VALUE_COL_IDX = 19

CALL_OI_COL_IDX = 1
PUT_OI_COL_IDX = 11
CALL_VOL_COL_IDX = 2
PUT_VOL_COL_IDX = 10

CALL_GAMMA_COL_IDX = 3
CALL_DELTA_COL_IDX = 4
CALL_VEGA_COL_IDX = 5
PUT_GAMMA_COL_IDX = 7
PUT_DELTA_COL_IDX = 8
PUT_VEGA_COL_IDX = 9

FACTOR = 100_000_000

# -------------------------------------------------
# LIVE PRICE
# -------------------------------------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    r = requests.get(API_BASE, timeout=10).json()["result"]
    return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])

prices = {a: get_delta_price(a) for a in ASSETS}

c1, c2 = st.columns(2)
c1.metric("BTC Price", f"{int(prices['BTC']):,}")
c2.metric("ETH Price", f"{int(prices['ETH']):,}")

# -------------------------------------------------
# PCR COLLECTION
# -------------------------------------------------
pcr_rows = []

# =================================================
# MAIN LOOP — BTC & ETH
# =================================================
for UNDERLYING in ASSETS:

    st.divider()
    st.subheader(f"{UNDERLYING} Analysis")

    CSV_PATH = f"data/{UNDERLYING}.csv"
    df_raw = pd.read_csv(CSV_PATH)

    df = pd.DataFrame({
        "strike_price": pd.to_numeric(df_raw.iloc[:, STRIKE_COL_IDX], errors="coerce"),
        "value": pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),
        "call_oi": pd.to_numeric(df_raw.iloc[:, CALL_OI_COL_IDX], errors="coerce"),
        "put_oi": pd.to_numeric(df_raw.iloc[:, PUT_OI_COL_IDX], errors="coerce"),
        "call_vol": pd.to_numeric(df_raw.iloc[:, CALL_VOL_COL_IDX], errors="coerce"),
        "put_vol": pd.to_numeric(df_raw.iloc[:, PUT_VOL_COL_IDX], errors="coerce"),
        "call_gamma": pd.to_numeric(df_raw.iloc[:, CALL_GAMMA_COL_IDX], errors="coerce"),
        "call_delta": pd.to_numeric(df_raw.iloc[:, CALL_DELTA_COL_IDX], errors="coerce"),
        "call_vega": pd.to_numeric(df_raw.iloc[:, CALL_VEGA_COL_IDX], errors="coerce"),
        "put_gamma": pd.to_numeric(df_raw.iloc[:, PUT_GAMMA_COL_IDX], errors="coerce"),
        "put_delta": pd.to_numeric(df_raw.iloc[:, PUT_DELTA_COL_IDX], errors="coerce"),
        "put_vega": pd.to_numeric(df_raw.iloc[:, PUT_VEGA_COL_IDX], errors="coerce"),
        "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5],
    }).dropna(subset=["strike_price", "timestamp"])

    # ---------------- TIME SELECTORS ----------------
    timestamps = rotated_time_sort(df["timestamp"].unique())
    col1, col2 = st.columns(2)
    with col1:
        t1 = st.selectbox(f"{UNDERLYING} Time 1 (Latest)", timestamps, index=0, key=f"{UNDERLYING}_t1")
    with col2:
        t2 = st.selectbox(f"{UNDERLYING} Time 2 (Previous)", timestamps, index=1, key=f"{UNDERLYING}_t2")

    # ---------------- PCR (CSV SNAPSHOT) ----------------
    def compute_pcr(d):
        return (
            d["put_oi"].sum() / d["call_oi"].sum(),
            d["put_vol"].sum() / d["call_vol"].sum(),
        )

    pcr_t1_oi, pcr_t1_vol = compute_pcr(df[df["timestamp"] == t1])
    pcr_t2_oi, pcr_t2_vol = compute_pcr(df[df["timestamp"] == t2])

    # ---------------- LIVE CHAIN ----------------
    df_live = pd.json_normalize(
        requests.get(
            f"{API_BASE}?contract_types=call_options,put_options"
            f"&underlying_asset_symbols={UNDERLYING}"
            f"&expiry_date={EXPIRY}"
        ).json()["result"]
    )

    df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")
    df_live["volume"] = pd.to_numeric(df_live["volume"], errors="coerce")

    pcr_live_oi = (
        df_live[df_live["contract_type"] == "put_options"]["oi_contracts"].sum()
        / df_live[df_live["contract_type"] == "call_options"]["oi_contracts"].sum()
    )

    pcr_live_vol = (
        df_live[df_live["contract_type"] == "put_options"]["volume"].sum()
        / df_live[df_live["contract_type"] == "call_options"]["volume"].sum()
    )

    pcr_rows.append([
        UNDERLYING,
        pcr_live_oi,
        pcr_t1_oi,
        pcr_t2_oi,
        pcr_live_vol,
        pcr_t1_vol,
        pcr_t2_vol,
    ])

    # -------------------------------------------------
    # 🔴 YOUR ORIGINAL MAX PAIN + GREEKS LOGIC CONTINUES
    # -------------------------------------------------
    # ⬇️ EVERYTHING FROM:
    # HISTORICAL MAX PAIN
    # LIVE MAX PAIN
    # GREEKS (TIME-1 vs LIVE)
    # FINAL MERGE
    # ATM HIGHLIGHT
    # DISPLAY
    # ⬇️ REMAINS **UNCHANGED**
    #
    # 👉 Paste that entire block here exactly as-is
    #
    # st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")
    # st.dataframe(...)
