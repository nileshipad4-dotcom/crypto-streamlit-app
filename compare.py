import streamlit as st
import pandas as pd
import requests
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}
EXPIRY = "15-12-2025"

STRIKE_COL_IDX = 6   # G:G
MAX_PAIN_COL_IDX = 19  # T:T

st.set_page_config(page_title="Live vs Historical Max Pain", layout="wide")
st.title("📊 Max Pain Comparison (Live vs BTC.csv)")

# -------------------------------------------------
# LIVE MAX PAIN (DELTA API)
# -------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_max_pain():

    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols=BTC"
        f"&expiry_date={EXPIRY}"
    )

    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    raw = pd.json_normalize(r.json().get("result", []))

    raw = raw[
        ["strike_price", "contract_type", "mark_price", "oi_contracts"]
    ].copy()

    raw["strike_price"] = pd.to_numeric(raw["strike_price"], errors="coerce")

    calls = raw[raw["contract_type"] == "call_options"].copy()
    puts  = raw[raw["contract_type"] == "put_options"].copy()

    calls = calls.rename(columns={
        "mark_price": "call_mark",
        "oi_contracts": "call_oi"
    }).drop(columns="contract_type")

    puts = puts.rename(columns={
        "mark_price": "put_mark",
        "oi_contracts": "put_oi"
    }).drop(columns="contract_type")

    df = pd.merge(calls, puts, on="strike_price", how="inner")
    df = df.sort_values("strike_price").reset_index(drop=True)

    A = df["call_mark"].astype(float).to_numpy()
    B = df["call_oi"].astype(float).to_numpy()
    G = df["strike_price"].astype(float).to_numpy()
    L = df["put_oi"].astype(float).to_numpy()
    M = df["put_mark"].astype(float).to_numpy()

    U = []
    for i in range(len(df)):
        Q = -np.sum(A[i:] * B[i:])
        R = G[i] * np.sum(B[:i]) - np.sum(G[:i] * B[:i])
        S = -np.sum(M[:i] * L[:i])
        T = np.sum(G[i:] * L[i:]) - G[i] * np.sum(L[i:])
        U.append(round((Q + R + S + T) / 10000))

    df["live_max_pain"] = U
    return df[["strike_price", "live_max_pain"]]


df_live = fetch_live_max_pain()

# -------------------------------------------------
# HISTORICAL MAX PAIN (BTC.csv — BY POSITION)
# -------------------------------------------------
btc_path = "data/BTC.csv"
df_hist = pd.read_csv(btc_path)

# Timestamp column (safe detection)
ts_col = next(
    (c for c in df_hist.columns if "timestamp" in c.lower()),
    None
)

if ts_col is None:
    st.error("timestamp_IST column not found in BTC.csv")
    st.stop()

# Extract required columns BY POSITION
df_hist_extracted = pd.DataFrame({
    "strike_price": pd.to_numeric(df_hist.iloc[:, STRIKE_COL_IDX], errors="coerce"),
    "historical_max_pain": pd.to_numeric(df_hist.iloc[:, MAX_PAIN_COL_IDX], errors="coerce"),
    "timestamp_IST": df_hist[ts_col]
})

# Timestamp selector
st.subheader("Select Historical Timestamp (BTC.csv)")
timestamps = sorted(df_hist_extracted["timestamp_IST"].dropna().unique())
selected_ts = st.selectbox("Timestamp (IST)", timestamps)

df_hist_snap = df_hist_extracted[
    df_hist_extracted["timestamp_IST"] == selected_ts
]

# -------------------------------------------------
# MERGE LIVE + HISTORICAL
# -------------------------------------------------
final_df = pd.merge(
    df_live,
    df_hist_snap[["strike_price", "historical_max_pain"]],
    on="strike_price",
    how="outer"
).sort_values("strike_price")

# -------------------------------------------------
# DISPLAY (ONLY 3 COLUMNS)
# -------------------------------------------------
st.subheader("Strike-wise Max Pain Comparison")

st.dataframe(
    final_df[
        ["strike_price", "live_max_pain", "historical_max_pain"]
    ],
    use_container_width=True
)

st.caption(
    f"Live max pain from Delta API • "
    f"Historical max pain from BTC.csv @ {selected_ts}"
)

