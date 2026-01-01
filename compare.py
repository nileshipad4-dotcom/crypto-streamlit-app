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
# AUTO REFRESH (60s)
# -------------------------------------------------
st_autorefresh(interval=60_000, key="auto_refresh")

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
EXPIRY = "01-01-2026"
ASSETS = ["BTC", "ETH"]

STRIKE_COL_IDX = 6
TIMESTAMP_COL_IDX = 14
VALUE_COL_IDX = 19

CALL_OI_COL_IDX = 1
PUT_OI_COL_IDX = 11
CALL_VOL_COL_IDX = 2
PUT_VOL_COL_IDX = 10

FACTOR = 100_000_000

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
# COMMON TIMESTAMP SELECTION
# -------------------------------------------------
df_ts = pd.read_csv("data/BTC.csv")
df_ts["timestamp"] = df_ts.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]

timestamps = rotated_time_sort(df_ts["timestamp"].unique())

col1, col2 = st.columns(2)
with col1:
    t1 = st.selectbox("Time 1 (Latest)", timestamps, index=0)
with col2:
    t2 = st.selectbox("Time 2 (Previous)", timestamps, index=1)

# -------------------------------------------------
# PCR COLLECTION
# -------------------------------------------------
pcr_rows = []

# =================================================
# MAIN LOOP — BTC & ETH
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
        "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5],
    }).dropna(subset=["strike_price", "timestamp"])

    # -------------------------------------------------
    # PCR (CSV SNAPSHOT)
    # -------------------------------------------------
    def compute_pcr(d):
        return (
            d["put_oi"].sum() / d["call_oi"].sum() if d["call_oi"].sum() else None,
            d["put_vol"].sum() / d["call_vol"].sum() if d["call_vol"].sum() else None,
        )

    pcr_t1_oi, pcr_t1_vol = compute_pcr(df[df["timestamp"] == t1])
    pcr_t2_oi, pcr_t2_vol = compute_pcr(df[df["timestamp"] == t2])

    # -------------------------------------------------
    # LIVE PCR
    # -------------------------------------------------
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
    # MAX PAIN COMPARISON
    # -------------------------------------------------
    df_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False)["value"].sum().rename(columns={"value": t1})
    df_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False)["value"].sum().rename(columns={"value": t2})

    merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
    merged["△ MP 2"] = merged[t1] - merged[t2]

    # ΔΔ MP 2 (strike-wise)
    merged["ΔΔ MP 2"] = -1 * (merged["△ MP 2"].shift(-1) - merged["△ MP 2"])

    # -------------------------------------------------
    # FINAL TABLE
    # -------------------------------------------------
    final = merged[
        ["strike_price", t1, t2, "△ MP 2", "ΔΔ MP 2"]
    ].round(0).astype("Int64")

    # -------------------------------------------------
    # ATM HIGHLIGHT
    # -------------------------------------------------
    atm_low = atm_high = None
    if prices[UNDERLYING]:
        strikes = final["strike_price"].astype(float).tolist()
        below = [s for s in strikes if s <= prices[UNDERLYING]]
        above = [s for s in strikes if s >= prices[UNDERLYING]]
        if below:
            atm_low = max(below)
        if above:
            atm_high = min(above)

    def highlight_atm(row):
        if row["strike_price"] in (atm_low, atm_high):
            return ["background-color:#000435"] * len(row)
        return [""] * len(row)

    # -------------------------------------------------
    # DISPLAY
    # -------------------------------------------------
    st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")
    st.dataframe(
        final.style.apply(highlight_atm, axis=1),
        use_container_width=True,
        height=650,
    )

# -------------------------------------------------
# PCR TABLES
# -------------------------------------------------
pcr_df = pd.DataFrame(
    pcr_rows,
    columns=[
        "Asset",
        "PCR OI (Live)",
        "PCR OI (T1)",
        "PCR OI (T2)",
        "PCR Vol (Live)",
        "PCR Vol (T1)",
        "PCR Vol (T2)",
    ],
).set_index("Asset")

st.subheader("📊 PCR Snapshot — Open Interest")
st.dataframe(
    pcr_df[["PCR OI (Live)", "PCR OI (T1)", "PCR OI (T2)"]].round(3),
    use_container_width=True,
)

st.subheader("📊 PCR Snapshot — Volume")
st.dataframe(
    pcr_df[["PCR Vol (Live)", "PCR Vol (T1)", "PCR Vol (T2)"]].round(3),
    use_container_width=True,
)

st.caption("🟡 ATM strikes highlighted | MP = Max Pain | △ = Time diff | ΔΔ = Strike diff")
