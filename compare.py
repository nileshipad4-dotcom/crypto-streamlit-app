# compare.py

import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# =================================================
# PATHS
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

BTC_PATH = os.path.join(DATA_DIR, "BTC.csv")
ETH_PATH = os.path.join(DATA_DIR, "ETH.csv")
TS_PATH = os.path.join(DATA_DIR, "timestamps.csv")

os.makedirs(DATA_DIR, exist_ok=True)

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
# AUTO TIMESTAMP MODE
# =================================================
auto_update_ts = st.checkbox("🔄 Auto-update timestamps", value=True)

# =================================================
# HELPERS
# =================================================
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

# =================================================
# TIMESTAMP MASTER (CORRECT – ALIGNED WITH COLLECTOR)
# =================================================
def update_timestamp_master():
    if not os.path.exists(BTC_PATH):
        st.error("❌ BTC.csv not found")
        return

    df = pd.read_csv(BTC_PATH)

    if "timestamp_IST" not in df.columns:
        st.error("❌ timestamp_IST column missing in BTC.csv")
        return

    timestamps = (
        df["timestamp_IST"]
        .astype(str)
        .dropna()
        .unique()
    )

    pd.DataFrame({"timestamp": timestamps}).to_csv(TS_PATH, index=False)

    st.caption(
        f"✅ timestamps.csv updated | rows={len(timestamps)} | latest={max(timestamps)}"
    )

# =================================================
# UPDATE TIMESTAMPS EVERY RUN
# =================================================
update_timestamp_master()

# =================================================
# LOAD TIMESTAMPS
# =================================================
if not os.path.exists(TS_PATH):
    st.stop()

df_ts = pd.read_csv(TS_PATH)
timestamps = rotated_time_sort(df_ts["timestamp"].tolist())

if len(timestamps) < 2:
    st.stop()

# =================================================
# TIMESTAMP SELECTION (STABLE)
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
def get_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])
    except:
        return None

prices = {a: get_price(a) for a in ASSETS}

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

    df = pd.read_csv(PATH)

    # PCR historical
    pcr_t1_oi = safe_ratio(
        df[df["timestamp_IST"] == t1]["put_oi"].sum(),
        df[df["timestamp_IST"] == t1]["call_oi"].sum(),
    )

    pcr_t2_oi = safe_ratio(
        df[df["timestamp_IST"] == t2]["put_oi"].sum(),
        df[df["timestamp_IST"] == t2]["call_oi"].sum(),
    )

    pcr_t1_vol = safe_ratio(
        df[df["timestamp_IST"] == t1]["put_volume"].sum(),
        df[df["timestamp_IST"] == t1]["call_volume"].sum(),
    )

    pcr_t2_vol = safe_ratio(
        df[df["timestamp_IST"] == t2]["put_volume"].sum(),
        df[df["timestamp_IST"] == t2]["call_volume"].sum(),
    )

    # LIVE PCR
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
        df_live[df_live["contract_type"] == "call_options"]["oi_contracts"].sum(),
    )

    pcr_live_vol = safe_ratio(
        df_live[df_live["contract_type"] == "put_options"]["volume"].sum(),
        df_live[df_live["contract_type"] == "call_options"]["volume"].sum(),
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

    # =================================================
    # MAX PAIN COMPARISON (HISTORICAL)
    # =================================================
    df_t1 = df[df["timestamp_IST"] == t1].groupby("strike_price", as_index=False)["max_pain"].sum()
    df_t2 = df[df["timestamp_IST"] == t2].groupby("strike_price", as_index=False)["max_pain"].sum()

    merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
    merged["△ MP"] = merged.iloc[:, 1] - merged.iloc[:, 2]

    st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")
    st.dataframe(merged.round(0), use_container_width=True)

# =================================================
# PCR TABLES
# =================================================
pcr_df = pd.DataFrame(
    pcr_rows,
    columns=[
        "Asset",
        "PCR OI (Current)",
        "PCR OI (T1)",
        "PCR OI (T2)",
        "PCR Vol (Current)",
        "PCR Vol (T1)",
        "PCR Vol (T2)",
    ],
).set_index("Asset")

st.subheader("📊 PCR Snapshot — OI")
st.dataframe(pcr_df[["PCR OI (Current)", "PCR OI (T1)", "PCR OI (T2)"]].round(3))

st.subheader("📊 PCR Snapshot — Volume")
st.dataframe(pcr_df[["PCR Vol (Current)", "PCR Vol (T1)", "PCR Vol (T2)"]].round(3))

st.caption("🟢 Source of truth: timestamp_IST from collector.py")
