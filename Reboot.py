# crypto compare — FINAL, CORRECT LIVE MP

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

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_hhmm():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

def safe_ratio(a, b):
    return a / b if b and not pd.isna(b) else None

def extract_fresh_timestamps_from_github(asset, pivot=PIVOT_TIME):
    df = pd.read_csv(f"{BASE_RAW_URL}{asset}.csv")
    times = (
        df.iloc[:, TIMESTAMP_COL_IDX]
        .astype(str)
        .str[:5]
        .dropna()
        .unique()
        .tolist()
    )

    pivot_m = int(pivot[:2]) * 60 + int(pivot[3:])
    def key(t):
        h, m = map(int, t.split(":"))
        return (pivot_m - (h * 60 + m)) % (24 * 60)

    return sorted(times, key=key)

# -------------------------------------------------
# ⏱ TIMESTAMP CONTROL (AS YOU WANT IT)
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
    st.session_state.timestamps = extract_fresh_timestamps_from_github(
        st.session_state.ts_asset
    )

timestamps = st.session_state.timestamps

t1 = st.selectbox(
    "Time 1 (Latest)",
    timestamps,
    index=0,
    key=f"t1_{st.session_state.run_id}",
)

t2 = st.selectbox(
    "Time 2 (Previous)",
    timestamps,
    index=1,
    key=f"t2_{st.session_state.run_id}",
)

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
# MAIN LOOP (ANALYTICS UNCHANGED, LIVE MP FIXED)
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

    # ---------------- LIVE MP (CORRECT MAX PAIN) ----------------
    df_live = pd.json_normalize(
        requests.get(
            f"{API_BASE}?contract_types=call_options,put_options"
            f"&underlying_asset_symbols={UNDERLYING}"
            f"&expiry_date={EXPIRY}",
            timeout=20,
        ).json()["result"]
    )

    df_mp = df_live[
        ["strike_price", "contract_type", "mark_price", "oi_contracts"]
    ].copy()

    for c in ["strike_price", "mark_price", "oi_contracts"]:
        df_mp[c] = pd.to_numeric(df_mp[c], errors="coerce")

    calls_mp = df_mp[df_mp["contract_type"] == "call_options"]
    puts_mp = df_mp[df_mp["contract_type"] == "put_options"]

    live_mp = pd.merge(
        calls_mp.rename(columns={
            "mark_price": "call_mark",
            "oi_contracts": "call_oi",
        }),
        puts_mp.rename(columns={
            "mark_price": "put_mark",
            "oi_contracts": "put_oi",
        }),
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

    now_ts = get_ist_hhmm()
    live_mp = live_mp.rename(columns={"Current": f"MP ({now_ts})"})

    # ---------------- FINAL MERGE ----------------
    final = merged.merge(live_mp, on="strike_price", how="left")

    final["△ MP 1"] = final[f"MP ({now_ts})"] - final[f"MP ({t1})"]
    final["ΔΔ MP 1"] = -1 * (
        final["△ MP 1"].shift(-1) - final["△ MP 1"]
    )

    final = final.sort_values("strike_price")

    final = final[
        [
            "strike_price",
            f"MP ({now_ts})",
            f"MP ({t1})",
            f"MP ({t2})",
            "△ MP 1",
            "△ MP 2",
            "ΔΔ MP 1",
        ]
    ].round(0).astype("Int64")

    final = final.reset_index(drop=True)

    # ---------------- HIGHLIGHTING ----------------
    atm = prices[UNDERLYING]
    min_mp = final[f"MP ({now_ts})"].min()

    def highlight(row):
        if row[f"MP ({now_ts})"] == min_mp:
            return ["background-color:#8B0000;color:white"] * len(row)
        if atm and row["strike_price"] == round(atm, -2):
            return ["background-color:#4B0082"] * len(row)
        return [""] * len(row)

    st.subheader(f"{UNDERLYING} — {t1} vs {t2}")
    st.dataframe(
        final.style.apply(highlight, axis=1),
        use_container_width=True,
        height=700,
    )

# -------------------------------------------------
# PCR TABLE
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
