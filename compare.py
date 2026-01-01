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

t1 = st.selectbox("Time 1 (Latest)", timestamps, index=0)
t2 = st.selectbox("Time 2 (Previous)", timestamps, index=1)

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

    # PCR
    def compute_pcr(d):
        return (
            d["put_oi"].sum() / d["call_oi"].sum() if d["call_oi"].sum() else None,
            d["put_vol"].sum() / d["call_vol"].sum() if d["call_vol"].sum() else None,
        )

    pcr_t1_oi, pcr_t1_vol = compute_pcr(df[df["timestamp"] == t1])
    pcr_t2_oi, pcr_t2_vol = compute_pcr(df[df["timestamp"] == t2])

    # -------------------------------------------------
    # LIVE OPTION CHAIN
    # -------------------------------------------------
    df_live = pd.json_normalize(
        requests.get(
            f"{API_BASE}?contract_types=call_options,put_options"
            f"&underlying_asset_symbols={UNDERLYING}"
            f"&expiry_date={EXPIRY}",
            timeout=20,
        ).json()["result"]
    )

    for c in ["strike_price", "mark_price", "oi_contracts", "volume"]:
        df_live[c] = pd.to_numeric(df_live[c], errors="coerce")

    pcr_rows.append([
        UNDERLYING,
        df_live[df_live["contract_type"] == "put_options"]["oi_contracts"].sum()
        / df_live[df_live["contract_type"] == "call_options"]["oi_contracts"].sum(),
        pcr_t1_oi,
        pcr_t2_oi,
        df_live[df_live["contract_type"] == "put_options"]["volume"].sum()
        / df_live[df_live["contract_type"] == "call_options"]["volume"].sum(),
        pcr_t1_vol,
        pcr_t2_vol,
    ])

    # -------------------------------------------------
    # HISTORICAL MP
    # -------------------------------------------------
    mp_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False).agg(**{t1: ("value", "sum")})
    mp_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False).agg(**{t2: ("value", "sum")})

    merged = mp_t1.merge(mp_t2, on="strike_price", how="outer")
    merged["Change"] = merged[t1] - merged[t2]

    # -------------------------------------------------
    # LIVE MAX PAIN (SAFE AGG)
    # -------------------------------------------------
    mp_live = (
        df_live.groupby("strike_price", as_index=False)
        .agg(Current=("mark_price", "mean"))
    )

    final = merged.merge(mp_live, on="strike_price", how="left")

    # -------------------------------------------------
    # DELTAS
    # -------------------------------------------------
    final["△ MP 1"] = final["Current"] - final[t1]
    final["△ MP 2"] = final["Change"]

    final["ΔΔ MP 1"] = -1 * (final["△ MP 1"].shift(-1) - final["△ MP 1"])
    final["ΔΔ MP 2"] = -1 * (final["△ MP 2"].shift(-1) - final["△ MP 2"])

    now_ts = get_ist_time()

    final = final.rename(columns={
        "Current": f"MP ({now_ts})",
        t1: f"MP ({t1})",
        t2: f"MP ({t2})",
    })

    final = final[
        [
            "strike_price",
            f"MP ({now_ts})",
            f"MP ({t1})",
            "△ MP 1",
            "ΔΔ MP 1",
            f"MP ({t2})",
            "△ MP 2",
            "ΔΔ MP 2",
        ]
    ].round(0)

    # -------------------------------------------------
    # ATM + LIVE MP HIGHLIGHT
    # -------------------------------------------------
    atm_low = atm_high = None
    if prices[UNDERLYING]:
        strikes = final["strike_price"].dropna().tolist()
        atm_low = max(s for s in strikes if s <= prices[UNDERLYING])
        atm_high = min(s for s in strikes if s >= prices[UNDERLYING])

    mp_col = f"MP ({now_ts})"
    mp_min = final[mp_col].min()

    def highlight(row):
        if row[mp_col] == mp_min:
            return ["background-color:#8B0000;color:white"] * len(row)
        if row["strike_price"] in (atm_low, atm_high):
            return ["background-color:#000435;color:white"] * len(row)
        return [""] * len(row)

    st.subheader(f"{UNDERLYING} — {t1} vs {t2}")
    st.dataframe(final.style.apply(highlight, axis=1), use_container_width=True, height=650)

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
st.dataframe(pcr_df[["PCR OI (Current)", "PCR OI (T1)", "PCR OI (T2)"]].round(3), use_container_width=True)

st.subheader("📊 PCR Snapshot — Volume")
st.dataframe(pcr_df[["PCR Vol (Current)", "PCR Vol (T1)", "PCR Vol (T2)"]].round(3), use_container_width=True)

st.caption("🟡 ATM band | 🔴 Live Max Pain | MP = Max Pain | △ = strike-wise diff")
