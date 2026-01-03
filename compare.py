# crypto compare

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

# -------------------------------------------------
# AUTO REFRESH
# -------------------------------------------------
refresh_count = st_autorefresh(interval=60_000, key="auto_refresh")

# -------------------------------------------------
# AUTO TIMESTAMP TOGGLE
# -------------------------------------------------
auto_update_ts = st.checkbox(
    "🔄 Auto-update timestamps on every refresh",
    value=True
)

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

def safe_ratio(a, b):
    return a / b if b and not pd.isna(b) else None

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
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

# -------------------------------------------------
# LIVE PRICE
# -------------------------------------------------
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

# -------------------------------------------------
# TIMESTAMP SELECTION (CORRECT ORDER)
# -------------------------------------------------
df_ts = pd.read_csv("data/BTC.csv")
df_ts["timestamp"] = df_ts.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]
timestamps = rotated_time_sort(df_ts["timestamp"].unique())

if "t1" not in st.session_state:
    st.session_state.t1 = timestamps[0]
if "t2" not in st.session_state:
    st.session_state.t2 = timestamps[1]

# 🔑 force update BEFORE widget creation
if auto_update_ts:
    st.session_state.t1 = timestamps[0]
    st.session_state.t2 = timestamps[1]

# 🔑 remount widgets
t1 = st.selectbox(
    "Time 1 (Latest)",
    timestamps,
    index=timestamps.index(st.session_state.t1),
    key=f"t1_{refresh_count}"
)

t2 = st.selectbox(
    "Time 2 (Previous)",
    timestamps,
    index=timestamps.index(st.session_state.t2),
    key=f"t2_{refresh_count}"
)

st.session_state.t1 = t1
st.session_state.t2 = t2

st.caption("🕒 Auto timestamps ON" if auto_update_ts else "🔒 Auto timestamps OFF")

# -------------------------------------------------
# PCR COLLECTION
# -------------------------------------------------
pcr_rows = []

# =================================================
# MAIN LOOP
# =================================================
for UNDERLYING in ASSETS:

    df_raw = pd.read_csv(f"data/{UNDERLYING}.csv")

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

    # LIVE CHAIN
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

    # ================= MAX PAIN =================
    df_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False)["value"].sum()
    df_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False)["value"].sum()

    merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
    merged["△ MP 2"] = merged.iloc[:, 1] - merged.iloc[:, 2]

    st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")
    st.dataframe(merged.round(0), use_container_width=True)

# -------------------------------------------------
# PCR TABLES
# -------------------------------------------------
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

st.caption("🟡 ATM band | 🔴 Live Max Pain | △ = Strike diff | ΔΔ = slope")
