import streamlit as st
import pandas as pd
import requests

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}
EXPIRY = "15-12-2025"

st.set_page_config(
    page_title="Live vs Historical Max Pain",
    layout="wide"
)

st.title("📊 Max Pain Comparison (Live vs BTC.csv)")

# -------------------------------------------------
# LIVE MAX PAIN (FROM DELTA API)
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

    # Keep only required columns
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

    # ---- MAX PAIN CALCULATION (LIVE) ----
    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    U = []
    for i in range(len(df)):
        Q = -sum(A[i:] * B[i:])
        R = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        S = -sum(M[:i] * L[:i])
        T = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        U.append(round((Q + R + S + T) / 10000))

    df["live_max_pain"] = U

    return df[["strike_price", "live_max_pain"]]


df_live = fetch_live_max_pain()

# -------------------------------------------------
# HISTORICAL MAX PAIN (FROM BTC.csv)
# -------------------------------------------------
btc_path = "data/BTC.csv"
df_hist = pd.read_csv(btc_path)

# Ensure correct dtypes
df_hist["strike_price"] = pd.to_numeric(df_hist["strike_price"], errors="coerce")
df_hist["max_pain"] = pd.to_numeric(df_hist["max_pain"], errors="coerce")

# Timestamp selector
st.subheader("Select Historical Timestamp (BTC.csv)")
timestamps = sorted(df_hist["timestamp_IST"].dropna().unique())
selected_ts = st.selectbox("Timestamp (IST)", timestamps)

df_hist_snap = df_hist[df_hist["timestamp_IST"] == selected_ts][
    ["strike_price", "max_pain"]
].rename(columns={"max_pain": "historical_max_pain"})

# -------------------------------------------------
# MERGE LIVE + HISTORICAL
# -------------------------------------------------
final_df = pd.merge(
    df_live,
    df_hist_snap,
    on="strike_price",
    how="outer"
).sort_values("strike_price")

# -------------------------------------------------
# DISPLAY (EXACTLY 3 COLUMNS)
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
