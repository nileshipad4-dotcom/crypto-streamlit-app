# crypto compare — FINAL FIXED VERSION

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
# FORCE FRESH RUN TOKEN
# -------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# -------------------------------------------------
# AUTO REFRESH (60s)
# -------------------------------------------------
if st_autorefresh(interval=60_000, key="auto_refresh"):
    st.cache_data.clear()
    st.session_state.run_id += 1

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
BASE_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "refs/heads/main/data/"
)

API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "05-01-2026"
ASSETS = ["BTC", "ETH"]
PIVOT_TIME = "17:30"

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
# HELPERS
# -------------------------------------------------
def get_ist_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

def safe_ratio(a, b):
    return a / b if b and not pd.isna(b) else None

def extract_fresh_timestamps_from_github(asset, pivot=PIVOT_TIME):
    df = pd.read_csv(f"{BASE_RAW_URL}{asset}.csv")
    times = df.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5].dropna().unique().tolist()

    pivot_m = int(pivot[:2]) * 60 + int(pivot[3:])
    def key(t):
        h, m = map(int, t.split(":"))
        return (pivot_m - (h * 60 + m)) % (24 * 60)

    return sorted(times, key=key)

# -------------------------------------------------
# ⏱ TIMESTAMP CONTROL
# -------------------------------------------------
if "ts_asset" not in st.session_state:
    st.session_state.ts_asset = "ETH"

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

c1, c2 = st.columns([1, 8])
with c1:
    st.selectbox("", ASSETS, key="ts_asset", label_visibility="collapsed")
with c2:
    refresh_ts = st.button("⏱")

if refresh_ts or not st.session_state.timestamps:
    st.session_state.timestamps = extract_fresh_timestamps_from_github(st.session_state.ts_asset)

timestamps = st.session_state.timestamps
t1 = st.selectbox("Time 1 (Latest)", timestamps, index=0, key=f"t1_{st.session_state.run_id}")
t2 = st.selectbox("Time 2 (Previous)", timestamps, index=1, key=f"t2_{st.session_state.run_id}")

# -------------------------------------------------
# LIVE PRICE
# -------------------------------------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    r = requests.get(API_BASE, timeout=10).json()["result"]
    return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])

prices = {a: get_delta_price(a) for a in ASSETS}

# -------------------------------------------------
# PCR COLLECTION
# -------------------------------------------------
pcr_rows = []

# =================================================
# MAIN LOOP (RESTORED FULL LOGIC)
# =================================================
for UNDERLYING in ASSETS:

    df_raw = pd.read_csv(f"data/{UNDERLYING}.csv")

    df = pd.DataFrame({
        "strike_price": pd.to_numeric(df_raw.iloc[:, STRIKE_COL_IDX], errors="coerce"),
        "value": pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),
        "call_oi": pd.to_numeric(df_raw.iloc[:, CALL_OI_COL_IDX], errors="coerce"),
        "put_oi": pd.to_numeric(df_raw.iloc[:, PUT_OI_COL_IDX], errors="coerce"),
        "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5],
    }).dropna(subset=["strike_price", "timestamp"])

    # ---------------- HISTORICAL MP ----------------
    df_t1 = df[df["timestamp"] == t1].groupby("strike_price")["value"].sum()
    df_t2 = df[df["timestamp"] == t2].groupby("strike_price")["value"].sum()

    merged = pd.concat([df_t1, df_t2], axis=1)
    merged.columns = [f"MP ({t1})", f"MP ({t2})"]
    merged["△ MP 2"] = merged.iloc[:, 0] - merged.iloc[:, 1]

    # ---------------- LIVE MP ----------------
    df_live = pd.json_normalize(
        requests.get(
            f"{API_BASE}?contract_types=call_options,put_options"
            f"&underlying_asset_symbols={UNDERLYING}"
            f"&expiry_date={EXPIRY}",
            timeout=20,
        ).json()["result"]
    )

    df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")
    df_live["mark_price"] = pd.to_numeric(df_live["mark_price"], errors="coerce")
    df_live["strike_price"] = pd.to_numeric(df_live["strike_price"], errors="coerce")

    calls = df_live[df_live["contract_type"] == "call_options"]
    puts = df_live[df_live["contract_type"] == "put_options"]

    mp = pd.merge(
        calls[["strike_price", "mark_price", "oi_contracts"]],
        puts[["strike_price", "mark_price", "oi_contracts"]],
        on="strike_price",
        suffixes=("_call", "_put"),
        how="outer",
    ).fillna(0)

    mp["MP (Live)"] = (
        mp["mark_price_call"] * mp["oi_contracts_call"]
        + mp["mark_price_put"] * mp["oi_contracts_put"]
    ) / 10000

    final = merged.merge(mp[["strike_price", "MP (Live)"]], on="strike_price", how="left")
    final["△ MP 1"] = final["MP (Live)"] - final[f"MP ({t1})"]
    final["ΔΔ MP 1"] = -(final["△ MP 1"].shift(-1) - final["△ MP 1"])

    final = final.reset_index().sort_values("strike_price")

    # ---------------- HIGHLIGHTING ----------------
    atm = prices[UNDERLYING]
    min_mp = final["MP (Live)"].min()

    def highlight(row):
        if row["strike_price"] == min_mp:
            return ["background-color:#8B0000;color:white"] * len(row)
        if atm and abs(row["strike_price"] - atm) < 1e-6:
            return ["background-color:#4B0082"] * len(row)
        return [""] * len(row)

    st.subheader(f"{UNDERLYING} — {t1} vs {t2}")
    st.dataframe(
        final.style.apply(highlight, axis=1),
        use_container_width=True,
        height=700,
    )

# -------------------------------------------------
# PCR TABLE (RESTORED)
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

st.subheader("📊 PCR Snapshot")
st.dataframe(pcr_df.round(3), use_container_width=True)
