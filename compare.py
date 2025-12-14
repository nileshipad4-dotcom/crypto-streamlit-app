import streamlit as st
import pandas as pd
import requests

st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Max Pain (BTC)")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BTC_PATH = "data/BTC.csv"

STRIKE_COL_IDX = 6      # G:G
VALUE_COL_IDX = 19     # column to compare (your Q/Q or chosen)
TIMESTAMP_COL_IDX = 14 # O:O

API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "15-12-2025"

# -------------------------------------------------
# LOAD BTC.csv
# -------------------------------------------------
df_raw = pd.read_csv(BTC_PATH)

df = pd.DataFrame({
    "strike_price": pd.to_numeric(df_raw.iloc[:, STRIKE_COL_IDX], errors="coerce"),
    "value": pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),
    "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX]
}).dropna(subset=["strike_price", "timestamp"])

# -------------------------------------------------
# TIME SELECTION
# -------------------------------------------------
timestamps = sorted(df["timestamp"].unique())

if len(timestamps) < 2:
    st.error("Not enough timestamps in BTC.csv")
    st.stop()

t1 = st.selectbox("Select Time 1", timestamps, index=0)
t2 = st.selectbox("Select Time 2", timestamps, index=1)

if t1 == t2:
    st.warning("Select two different timestamps")
    st.stop()

# -------------------------------------------------
# DATA FOR BOTH TIMES
# -------------------------------------------------
df_t1 = (
    df[df["timestamp"] == t1]
    .groupby("strike_price", as_index=False)["value"]
    .sum()
    .rename(columns={"value": "value_time_1"})
)

df_t2 = (
    df[df["timestamp"] == t2]
    .groupby("strike_price", as_index=False)["value"]
    .sum()
    .rename(columns={"value": "value_time_2"})
)

merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
merged["change"] = merged["value_time_2"] - merged["value_time_1"]

# -------------------------------------------------
# 🔥 FETCH LIVE OPTION CHAIN
# -------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_chain():
    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols=BTC"
        f"&expiry_date={EXPIRY}"
    )
    r = requests.get(url, timeout=20).json()["result"]
    return pd.json_normalize(r)

df_live_raw = fetch_live_chain()

# -------------------------------------------------
# FORMAT LIVE DATA
# -------------------------------------------------
df_live = df_live_raw[[
    "strike_price", "contract_type",
    "mark_price", "oi_contracts"
]].copy()

df_live["strike_price"] = pd.to_numeric(df_live["strike_price"], errors="coerce")
df_live["mark_price"] = pd.to_numeric(df_live["mark_price"], errors="coerce")
df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")

calls = df_live[df_live["contract_type"] == "call_options"]
puts  = df_live[df_live["contract_type"] == "put_options"]

live = pd.merge(
    calls.rename(columns={"mark_price":"call_mark","oi_contracts":"call_oi"}),
    puts.rename(columns={"mark_price":"put_mark","oi_contracts":"put_oi"}),
    on="strike_price",
    how="outer"
)

# -------------------------------------------------
# 🔥 COMPUTE LIVE MAX PAIN
# -------------------------------------------------
def compute_max_pain(df):
    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    mp = []
    for i in range(len(df)):
        Q = -sum(A[i:] * B[i:])
        R = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        S = -sum(M[:i] * L[:i])
        T = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        mp.append(round((Q + R + S + T) / 10000))

    df["max_pain"] = mp
    return df[["strike_price", "max_pain"]]

df_live_mp = compute_max_pain(live)

# -------------------------------------------------
# 🔥 MERGE LIVE MAX PAIN (LAST COLUMN)
# -------------------------------------------------
final = pd.merge(
    merged,
    df_live_mp,
    on="strike_price",
    how="left"
).sort_values("strike_price")

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
def color_change(v):
    if pd.isna(v): return ""
    if v > 0: return "background-color: lightgreen"
    if v < 0: return "background-color: lightcoral"
    return ""

st.subheader(f"Comparison {t1} → {t2} + Live Max Pain")

st.dataframe(
    final.style.applymap(color_change, subset=["change"]),
    use_container_width=True
)

st.caption("🟢 Increase | 🔴 Decrease | Live Max Pain from Delta Exchange")
