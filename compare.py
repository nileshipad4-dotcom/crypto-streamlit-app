# crypto_compare.py

import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# =================================================
# PATH SETUP (CRITICAL – DO NOT SKIP)
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

BTC_PATH = os.path.join(DATA_DIR, "BTC.csv")
ETH_PATH = os.path.join(DATA_DIR, "ETH.csv")
TS_PATH = os.path.join(DATA_DIR, "timestamps.csv")

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

# =================================================
# AUTO REFRESH
# =================================================
st_autorefresh(interval=60_000, key="auto_refresh")

# =================================================
# AUTO MODE TOGGLE
# =================================================
auto_update_ts = st.checkbox("🔄 Auto-update timestamps", value=True)

# =================================================
# HELPERS
# =================================================
def get_ist_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

def rotated_time_sort(times, pivot="17:30"):
    pivot_minutes = int(pivot[:2]) * 60 + int(pivot[3:])
    def key(t):
        h, m = map(int, t.split(":"))
        return ((h * 60 + m) - pivot_minutes) % (24 * 60)
    return sorted(times, key=key, reverse=True)

def safe_ratio(a, b):
    return a / b if b and not pd.isna(b) and b != 0 else None

# =================================================
# CONFIG
# =================================================
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "03-01-2026"
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

# =================================================
# TIMESTAMP MASTER UPDATER (THIS IS THE KEY)
# =================================================
def update_timestamp_master():
    if not os.path.exists(BTC_PATH):
        st.error(f"BTC.csv not found at {BTC_PATH}")
        return

    df = pd.read_csv(BTC_PATH)

    timestamps = (
        df.iloc[:, TIMESTAMP_COL_IDX]
        .astype(str)
        .str[:5]
        .dropna()
        .unique()
    )

    ts_df = pd.DataFrame({"timestamp": timestamps})
    ts_df.to_csv(TS_PATH, index=False)

    # PROOF ON SCREEN
    st.caption(
        f"✅ timestamps.csv updated | count={len(ts_df)} | latest={max(timestamps)}"
    )

# 🔍 DEBUG BTC CSV
df_debug = pd.read_csv(BTC_PATH)

st.write("📄 BTC.csv columns:", list(df_debug.columns))
st.write("📄 BTC.csv shape:", df_debug.shape)

# show first 5 rows of the supposed timestamp column
st.write(
    "🕒 Raw timestamp column preview:",
    df_debug.iloc[:, TIMESTAMP_COL_IDX].head(10)
)

# =================================================
# UPDATE TIMESTAMPS ON EVERY RUN
# =================================================
update_timestamp_master()

# =================================================
# LOAD TIMESTAMPS (SINGLE SOURCE OF TRUTH)
# =================================================
if not os.path.exists(TS_PATH):
    st.stop()

df_ts = pd.read_csv(TS_PATH)
timestamps = rotated_time_sort(df_ts["timestamp"].tolist())

if len(timestamps) < 2:
    st.stop()

# =================================================
# TIMESTAMP SELECTION (STABLE BY DESIGN)
# =================================================
if auto_update_ts:
    t1 = timestamps[0]
    t2 = timestamps[1]
    st.info(f"🕒 Auto mode: {t1} vs {t2}")
else:
    t1 = st.selectbox("Time 1 (Latest)", timestamps, index=0)
    t2 = st.selectbox("Time 2 (Previous)", timestamps, index=1)

# =================================================
# LIVE PRICE
# =================================================
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])
    except Exception:
        return None

prices = {a: get_delta_price(a) for a in ASSETS}

c1, c2 = st.columns(2)
c1.metric("BTC Price", f"{int(prices['BTC']):,}" if prices["BTC"] else "Error")
c2.metric("ETH Price", f"{int(prices['ETH']):,}" if prices["ETH"] else "Error")

# =================================================
# PCR COLLECTION
# =================================================
pcr_rows = []

# =================================================
# MAIN LOOP
# =================================================
for UNDERLYING, PATH in zip(ASSETS, [BTC_PATH, ETH_PATH]):

    df_raw = pd.read_csv(PATH)

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

    # PCR historical
    pcr_t1_oi = safe_ratio(df[df["timestamp"] == t1]["put_oi"].sum(),
                           df[df["timestamp"] == t1]["call_oi"].sum())
    pcr_t2_oi = safe_ratio(df[df["timestamp"] == t2]["put_oi"].sum(),
                           df[df["timestamp"] == t2]["call_oi"].sum())
    pcr_t1_vol = safe_ratio(df[df["timestamp"] == t1]["put_vol"].sum(),
                            df[df["timestamp"] == t1]["call_vol"].sum())
    pcr_t2_vol = safe_ratio(df[df["timestamp"] == t2]["put_vol"].sum(),
                            df[df["timestamp"] == t2]["call_vol"].sum())

    # Live chain
    df_live = pd.json_normalize(
        requests.get(
            f"{API_BASE}?contract_types=call_options,put_options"
            f"&underlying_asset_symbols={UNDERLYING}"
            f"&expiry_date={EXPIRY}",
            timeout=20
        ).json()["result"]
    )

    df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")
    df_live["volume"] = pd.to_numeric(df_live["volume"], errors="coerce")

    pcr_live_oi = safe_ratio(
        df_live[df_live["contract_type"] == "put_options"]["oi_contracts"].sum(),
        df_live[df_live["contract_type"] == "call_options"]["oi_contracts"].sum()
    )

    pcr_live_vol = safe_ratio(
        df_live[df_live["contract_type"] == "put_options"]["volume"].sum(),
        df_live[df_live["contract_type"] == "call_options"]["volume"].sum()
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

    # MAX PAIN (historical)
    df_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False)["value"].sum()
    df_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False)["value"].sum()

    merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
    merged["△ MP 2"] = merged.iloc[:, 1] - merged.iloc[:, 2]

