import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Data (BTC)")

# =================================================
# AUTO REFRESH (30s)
# =================================================
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30 * 1000, key="refresh")
except:
    pass

# =================================================
# CONFIG
# =================================================
BTC_PATH = "data/BTC.csv"

STRIKE_COL_IDX = 6
VALUE_COL_IDX = 19
TIMESTAMP_COL_IDX = 14

API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "15-12-2025"

# =================================================
# IST TIME
# =================================================
def get_ist_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M:%S")

current_time_label = get_ist_time()

# =================================================
# LIVE PRICE
# =================================================
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        sym = "BTCUSD" if symbol == "BTC" else "ETHUSD"
        return float(next(x for x in r if x["symbol"] == sym)["mark_price"])
    except:
        return None

price_btc = get_delta_price("BTC")
price_eth = get_delta_price("ETH")

st.sidebar.metric("BTC Price (Delta)", f"{int(price_btc):,}" if price_btc else "Error")
st.sidebar.metric("ETH Price (Delta)", f"{int(price_eth):,}" if price_eth else "Error")

# =================================================
# LOAD BTC.csv
# =================================================
df_raw = pd.read_csv(BTC_PATH)

df = pd.DataFrame({
    "strike_price": pd.to_numeric(df_raw.iloc[:, STRIKE_COL_IDX], errors="coerce"),
    "value": pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),
    "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX]
}).dropna(subset=["strike_price", "timestamp"])

# =================================================
# TIME SELECTION
# =================================================
timestamps = sorted(df["timestamp"].unique())

if len(timestamps) < 2:
    st.error("Not enough timestamps in BTC.csv")
    st.stop()

t1 = st.selectbox("Select Time 1", timestamps, index=0)
t2 = st.selectbox("Select Time 2", timestamps, index=1)

if t1 == t2:
    st.warning("Select two different timestamps")
    st.stop()

# =================================================
# DATA FOR BOTH TIMES
# =================================================
df_t1 = (
    df[df["timestamp"] == t1]
    .groupby("strike_price", as_index=False)["value"]
    .sum()
    .rename(columns={"value": t1})
)

df_t2 = (
    df[df["timestamp"] == t2]
    .groupby("strike_price", as_index=False)["value"]
    .sum()
    .rename(columns={"value": t2})
)

merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")

merged["change"] = merged[t1] - merged[t2]

# =================================================
# FETCH LIVE OPTION CHAIN
# =================================================
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

# =================================================
# FORMAT LIVE DATA
# =================================================
df_live = df_live_raw[
    ["strike_price", "contract_type", "mark_price", "oi_contracts"]
].copy()

for c in ["strike_price", "mark_price", "oi_contracts"]:
    df_live[c] = pd.to_numeric(df_live[c], errors="coerce")

calls = df_live[df_live["contract_type"] == "call_options"]
puts  = df_live[df_live["contract_type"] == "put_options"]

live = pd.merge(
    calls.rename(columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}),
    puts.rename(columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}),
    on="strike_price",
    how="outer"
)

# =================================================
# COMPUTE LIVE VALUE (MAX PAIN LOGIC)
# =================================================
def compute_live_value(df):
    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    out = []
    for i in range(len(df)):
        Q = -sum(A[i:] * B[i:])
        R = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        S = -sum(M[:i] * L[:i])
        T = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        out.append(round((Q + R + S + T) / 10000))

    df[current_time_label] = out
    return df[["strike_price", current_time_label]]

df_live_value = compute_live_value(live)

# =================================================
# MERGE ALL
# =================================================
final = pd.merge(
    merged,
    df_live_value,
    on="strike_price",
    how="left"
).sort_values("strike_price")

final[f"{current_time_label} - {t1}"] = final[current_time_label] - final[t1]

# =================================================
# COLUMN ORDER
# =================================================
final = final[
    [
        "strike_price",
        current_time_label,
        f"{current_time_label} - {t1}",
        t1,
        t2,
        "change",
    ]
]

# =================================================
# INTEGER DISPLAY
# =================================================
for col in final.columns:
    final[col] = final[col].round(0).astype("Int64")

# =================================================
# STYLING
# =================================================
def color_change(v):
    if pd.isna(v): return ""
    if v > 0: return "background-color: lightgreen"
    if v < 0: return "background-color: lightcoral"
    return ""

st.subheader(f"Live vs {t1} vs {t2}")

st.dataframe(
    final.style.applymap(
        color_change,
        subset=[f"{current_time_label} - {t1}", "change"]
    ),
    use_container_width=True
)

st.caption(
    "🟢 Positive | 🔴 Negative | "
    "Live column shows current time snapshot • Auto-refresh 30s"
)
