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

def safe_ratio(a, b):
    return a / b if b and not pd.isna(b) else None

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
# TIME SELECTION
# -------------------------------------------------
df_ts = pd.read_csv("data/BTC.csv")
df_ts["timestamp"] = df_ts.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]
timestamps = rotated_time_sort(df_ts["timestamp"].unique())

t1 = st.selectbox("Time 1 (Latest)", timestamps, index=0)
t2 = st.selectbox("Time 2 (Previous)", timestamps, index=1)

# -------------------------------------------------
# PCR STORAGE
# -------------------------------------------------
pcr_rows = []

st.divider()
st.header("📈 MAX PAIN TABLES")

# =================================================
# MAIN LOOP
# =================================================
for UNDERLYING in ASSETS:

    with st.container():  # 🔑 THIS FIXES VISIBILITY

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

        # PCR CSV
        pcr_t1_oi = safe_ratio(
            df[df["timestamp"] == t1]["put_oi"].sum(),
            df[df["timestamp"] == t1]["call_oi"].sum(),
        )
        pcr_t2_oi = safe_ratio(
            df[df["timestamp"] == t2]["put_oi"].sum(),
            df[df["timestamp"] == t2]["call_oi"].sum(),
        )
        pcr_t1_vol = safe_ratio(
            df[df["timestamp"] == t1]["put_vol"].sum(),
            df[df["timestamp"] == t1]["call_vol"].sum(),
        )
        pcr_t2_vol = safe_ratio(
            df[df["timestamp"] == t2]["put_vol"].sum(),
            df[df["timestamp"] == t2]["call_vol"].sum(),
        )

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
            df_live.loc[df_live["contract_type"] == "put_options", "oi_contracts"].sum(),
            df_live.loc[df_live["contract_type"] == "call_options", "oi_contracts"].sum(),
        )

        pcr_live_vol = safe_ratio(
            df_live.loc[df_live["contract_type"] == "put_options", "volume"].sum(),
            df_live.loc[df_live["contract_type"] == "call_options", "volume"].sum(),
        )

        pcr_rows.append([
            UNDERLYING,
            pcr_live_oi, pcr_t1_oi, pcr_t2_oi,
            pcr_live_vol, pcr_t1_vol, pcr_t2_vol
        ])

        # HISTORICAL MP
        df_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False)["value"].sum()
        df_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False)["value"].sum()

        merged = df_t1.merge(df_t2, on="strike_price", how="outer", suffixes=(f" ({t1})", f" ({t2})"))
        merged["△ MP 2"] = merged.iloc[:, 1] - merged.iloc[:, 2]

        # LIVE MP
        mp = df_live[["strike_price", "contract_type", "mark_price", "oi_contracts"]].copy()
        mp[["strike_price","mark_price","oi_contracts"]] = mp[["strike_price","mark_price","oi_contracts"]].apply(pd.to_numeric, errors="coerce")

        calls = mp[mp["contract_type"] == "call_options"]
        puts = mp[mp["contract_type"] == "put_options"]

        live = calls.merge(
            puts, on="strike_price", how="outer", suffixes=("_call","_put")
        ).sort_values("strike_price")

        live["MP LIVE"] = (
            (live["strike_price"] * live["oi_contracts_put"].fillna(0))
            - (live["strike_price"] * live["oi_contracts_call"].fillna(0))
        ).abs()

        final = merged.merge(live[["strike_price","MP LIVE"]], on="strike_price", how="left")

        now = get_ist_time()
        final = final.rename(columns={"MP LIVE": f"MP ({now})"})

        mp_col = f"MP ({now})"
        min_mp = final[mp_col].min()

        atm_low = atm_high = None
        if prices[UNDERLYING]:
            strikes = final["strike_price"].tolist()
            atm_low = max([s for s in strikes if s <= prices[UNDERLYING]], default=None)
            atm_high = min([s for s in strikes if s >= prices[UNDERLYING]], default=None)

        def highlight(row):
            if row[mp_col] == min_mp:
                return ["background-color:#8B0000;color:white"] * len(row)
            if row["strike_price"] in (atm_low, atm_high):
                return ["background-color:#000435;color:white"] * len(row)
            return [""] * len(row)

        st.subheader(f"{UNDERLYING} — Max Pain")
        st.dataframe(final.style.apply(highlight, axis=1), use_container_width=True)

# -------------------------------------------------
# PCR TABLES
# -------------------------------------------------
st.divider()
st.header("📊 PCR SNAPSHOT")

pcr_df = pd.DataFrame(
    pcr_rows,
    columns=[
        "Asset",
        "PCR OI (Current)", "PCR OI (T1)", "PCR OI (T2)",
        "PCR Vol (Current)", "PCR Vol (T1)", "PCR Vol (T2)",
    ],
).set_index("Asset")

st.subheader("PCR — Open Interest")
st.dataframe(pcr_df[["PCR OI (Current)", "PCR OI (T1)", "PCR OI (T2)"]].round(3))

st.subheader("PCR — Volume")
st.dataframe(pcr_df[["PCR Vol (Current)", "PCR Vol (T1)", "PCR Vol (T2)"]].round(3))

st.caption("🔴 Live Max Pain | 🔵 ATM | △ = diff | ΔΔ = slope")
