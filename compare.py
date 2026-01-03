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

def safe_ratio(a, b):
    return a / b if b and not pd.isna(b) and b != 0 else None

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "03-01-2026"
ASSETS = ["BTC", "ETH"]

# column indices (kept as-is for rest of logic)
STRIKE_COL_IDX = 6
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
    except:
        return None

prices = {a: get_delta_price(a) for a in ASSETS}

c1, c2 = st.columns(2)
c1.metric("BTC Price", f"{int(prices['BTC']):,}" if prices["BTC"] else "Error")
c2.metric("ETH Price", f"{int(prices['ETH']):,}" if prices["ETH"] else "Error")

# -------------------------------------------------
# TIMESTAMP RELOAD (FROM BTC.csv → timestamp_IST)
# -------------------------------------------------
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

reload_ts = st.button("🔁 Reload timestamps from BTC.csv")

if reload_ts or not st.session_state.timestamps:
    try:
        df_ts = pd.read_csv("data/BTC.csv")

        if "timestamp_IST" not in df_ts.columns:
            st.error("❌ timestamp_IST column not found in BTC.csv")
            st.stop()

        st.session_state.timestamps = rotated_time_sort(
            df_ts["timestamp_IST"]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )

        st.success(f"Loaded {len(st.session_state.timestamps)} timestamps")

    except Exception as e:
        st.error(f"Failed to reload timestamps: {e}")
        st.stop()

timestamps = st.session_state.timestamps

if len(timestamps) < 2:
    st.warning("Not enough timestamps available")
    st.stop()

# -------------------------------------------------
# TIMESTAMP SELECTION
# -------------------------------------------------
t1 = st.selectbox(
    "Time 1 (Latest)",
    timestamps,
    index=0,
    key="t1_select"
)

t2 = st.selectbox(
    "Time 2 (Previous)",
    timestamps,
    index=1,
    key="t2_select"
)

# -------------------------------------------------
# PCR COLLECTION
# -------------------------------------------------
pcr_rows = []

# =================================================
# MAIN LOOP
# =================================================
for UNDERLYING in ASSETS:

    df_raw = pd.read_csv(f"data/{UNDERLYING}.csv")

    if "timestamp_IST" not in df_raw.columns:
        st.error(f"❌ timestamp_IST missing in {UNDERLYING}.csv")
        continue

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
        "timestamp": df_raw["timestamp_IST"].astype(str),
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
        None,
        None,
    ])

    # -------------------------------------------------
    # HISTORICAL MAX PAIN
    # -------------------------------------------------
    df_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False)["value"].sum()
    df_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False)["value"].sum()

    merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
    merged["△ MP 2"] = merged.iloc[:, 1] - merged.iloc[:, 2]

    # -------------------------------------------------
    # LIVE MAX PAIN
    # -------------------------------------------------
    df_mp = df_live[["strike_price", "contract_type", "mark_price", "oi_contracts"]].copy()
    for c in ["strike_price", "mark_price", "oi_contracts"]:
        df_mp[c] = pd.to_numeric(df_mp[c], errors="coerce")

    calls_mp = df_mp[df_mp["contract_type"] == "call_options"]
    puts_mp = df_mp[df_mp["contract_type"] == "put_options"]

    live_mp = pd.merge(
        calls_mp.rename(columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}),
        puts_mp.rename(columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}),
        on="strike_price",
        how="outer",
    ).sort_values("strike_price")

    def compute_max_pain(df):
        A = df["call_mark"].fillna(0).values
        B = df["call_oi"].fillna(0).values
        G = df["strike_price"].values
        L = df["put_oi"].fillna(0).values
        M = df["put_mark"].fillna(0).values

        df["Current"] = [
            round(
                (
                    -sum(A[i:] * B[i:])
                    + G[i] * sum(B[:i])
                    - sum(G[:i] * B[:i])
                    - sum(M[:i] * L[:i])
                    + sum(G[i:] * L[i:])
                    - G[i] * sum(L[i:])
                ) / 10000
            )
            for i in range(len(df))
        ]
        return df[["strike_price", "Current"]]

    live_mp = compute_max_pain(live_mp)

    final = merged.merge(live_mp, on="strike_price", how="left")

    final["△ MP 1"] = final["Current"] - final.iloc[:, 1]
    final["ΔΔ MP 1"] = -1 * (final["△ MP 1"].shift(-1) - final["△ MP 1"])
    final["ΔΔ MP 2"] = -1 * (final["△ MP 2"].shift(-1) - final["△ MP 2"])

    now_ts = get_ist_time()

    final = final.rename(columns={
        "Current": f"MP ({now_ts})",
        "△ MP 1": "△ MP 1",
        "△ MP 2": "△ MP 2",
    })

    final = final[
        [
            "strike_price",
            f"MP ({now_ts})",
            "△ MP 1",
            "△ MP 2",
            "ΔΔ MP 1",
            "ΔΔ MP 2",
        ]
    ].round(0).astype("Int64")

    st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")
    st.dataframe(final, use_container_width=True, height=700)

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
