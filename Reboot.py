# crypto compare — time-aligned CSV logic (NO CACHE, NO STALE STATE)

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
# AUTO REFRESH (60s)
# -------------------------------------------------
st_autorefresh(interval=60_000, key="auto_refresh")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_ist_time_str(dt=None):
    if dt is None:
        dt = get_ist_datetime()
    return dt.strftime("%H:%M")

def rotated_time_sort(times, pivot="17:30"):
    pivot_minutes = int(pivot[:2]) * 60 + int(pivot[3:])
    def key(t):
        h, m = map(int, t.split(":"))
        return ((h * 60 + m) - pivot_minutes) % (24 * 60)
    return sorted(times, key=key, reverse=True)

def safe_ratio(a, b):
    return a / b if b and not pd.isna(b) else None

def find_latest_matching_times(csv_times, now_dt):
    """
    Walk backward minute-by-minute until a timestamp is found.
    Returns (t1, t2)
    """
    csv_set = set(csv_times)

    probe = now_dt.replace(second=0, microsecond=0)
    for _ in range(180):  # search back up to 3 hours
        probe_str = probe.strftime("%H:%M")
        if probe_str in csv_set:
            sorted_times = rotated_time_sort(csv_times)
            idx = sorted_times.index(probe_str)
            t2 = sorted_times[idx + 1] if idx + 1 < len(sorted_times) else None
            return probe_str, t2
        probe -= timedelta(minutes=1)

    return None, None

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "05-01-2026"
ASSETS = ["BTC", "ETH"]

BTC_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "refs/heads/main/data/BTC.csv"
)

ETH_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "refs/heads/main/data/ETH.csv"
)

CSV_URLS = {
    "BTC": BTC_CSV_URL,
    "ETH": ETH_CSV_URL,
}

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
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])
    except:
        return None

prices = {a: get_delta_price(a) for a in ASSETS}

c1, c2 = st.columns(2)
c1.metric("BTC Price", f"{int(prices['BTC']):,}" if prices["BTC"] else "Error")
c2.metric("ETH Price", f"{int(prices['ETH']):,}" if prices["ETH"] else "Error")

# -------------------------------------------------
# READ CSV FRESH (EVERY RUN)
# -------------------------------------------------
df_ts = pd.read_csv(BTC_CSV_URL)
df_ts["timestamp"] = df_ts.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]
all_times = df_ts["timestamp"].dropna().unique().tolist()

now_ist = get_ist_datetime()
t1, t2 = find_latest_matching_times(all_times, now_ist)

if not t1 or not t2:
    st.error("No valid timestamps found in CSV")
    st.stop()

st.caption(f"🕒 Auto-selected timestamps → T1: {t1} | T2: {t2}")

# -------------------------------------------------
# PCR COLLECTION
# -------------------------------------------------
pcr_rows = []

# =================================================
# MAIN LOOP
# =================================================
for UNDERLYING in ASSETS:

    df_raw = pd.read_csv(CSV_URLS[UNDERLYING])

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
    pcr_t1_oi = safe_ratio(
        df[df["timestamp"] == t1]["put_oi"].sum(),
        df[df["timestamp"] == t1]["call_oi"].sum(),
    )

    pcr_t2_oi = safe_ratio(
        df[df["timestamp"] == t2]["put_oi"].sum(),
        df[df["timestamp"] == t2]["call_oi"].sum(),
    )

    # ---------------- LIVE CHAIN ----------------
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
        df_live.loc[df_live["contract_type"] == "put_options", "oi_contracts"].sum(),
        df_live.loc[df_live["contract_type"] == "call_options", "oi_contracts"].sum(),
    )

    pcr_rows.append([
        UNDERLYING,
        pcr_live_oi,
        pcr_t1_oi,
        pcr_t2_oi,
    ])

    # -------------------------------------------------
    # DISPLAY PCR
    # -------------------------------------------------
    st.subheader(f"{UNDERLYING} PCR")
    st.write({
        "PCR OI (Live)": pcr_live_oi,
        f"PCR OI ({t1})": pcr_t1_oi,
        f"PCR OI ({t2})": pcr_t2_oi,
    })

st.caption("✅ CSV read fresh every run | timestamps auto-aligned to clock")
