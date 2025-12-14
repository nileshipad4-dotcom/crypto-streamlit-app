import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

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
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "15-12-2025"

STRIKE_COL_IDX = 6
VALUE_COL_IDX = 19
TIMESTAMP_COL_IDX = 14

# =================================================
# UNDERLYING TOGGLE
# =================================================
underlying = st.sidebar.selectbox("Underlying", ["BTC", "ETH"])
DATA_PATH = f"data/{underlying}.csv"

# =================================================
# IST TIME (HH:MM ONLY)
# =================================================
def get_ist_time_hhmm():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

current_time_label = get_ist_time_hhmm()

# =================================================
# LIVE PRICE
# =================================================
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        sym = f"{symbol}USD"
        return float(next(x for x in r if x["symbol"] == sym)["mark_price"])
    except:
        return None

price = get_delta_price(underlying)

st.sidebar.metric(
    f"{underlying} Price (Delta)",
    f"{int(price):,}" if price else "Error"
)

# =================================================
# LOAD CSV
# =================================================
df_raw = pd.read_csv(DATA_PATH)

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
    st.error("Not enough timestamps")
    st.stop()

t1 = st.selectbox("Select Time 1", timestamps, index=0)
t2 = st.selectbox("Select Time 2", timestamps, index=1)

if t1 == t2:
    st.warning("Select two different timestamps")
    st.stop()

# =================================================
# HISTORICAL DATA
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
# LIVE OPTION CHAIN
# =================================================
@st.cache_data(ttl=30)
def fetch_live_chain():
    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={EXPIRY}"
    )
    r = requests.get(url, timeout=20).json()["result"]
    return pd.json_normalize(r)

df_live_raw = fetch_live_chain()

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
# LIVE CALCULATION (MAX PAIN LOGIC)
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

df_live_val = compute_live_value(live)

# =================================================
# FINAL MERGE
# =================================================
final = pd.merge(
    merged,
    df_live_val,
    on="strike_price",
    how="left"
).sort_values("strike_price")

final[f"{current_time_label} - {t1}"] = final[current_time_label] - final[t1]

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

for c in final.columns:
    final[c] = final[c].round(0).astype("Int64")

# =================================================
# ATM STRIKE DETECTION
# =================================================
lower_strike = upper_strike = None

if price:
    strikes = final["strike_price"].dropna().astype(float).tolist()
    lower = [s for s in strikes if s <= price]
    upper = [s for s in strikes if s >= price]
    if lower:
        lower_strike = max(lower)
    if upper:
        upper_strike = min(upper)

# =================================================
# STYLING
# =================================================
def color_change(v):
    if pd.isna(v): return ""
    if v > 0: return "background-color: lightgreen"
    if v < 0: return "background-color: lightcoral"
    return ""

def highlight_atm(row):
    styles = [""] * len(row)
    if row["strike_price"] in (lower_strike, upper_strike):
        styles[0] = "background-color: #ffd6e8"  # light pink
    return styles

# =================================================
# DISPLAY
# =================================================
st.subheader(f"{underlying} — Live ({current_time_label}) vs {t1} vs {t2}")

st.dataframe(
    final.style
        .applymap(color_change, subset=["change", f"{current_time_label} - {t1}"])
        .apply(highlight_atm, axis=1),
    use_container_width=True
)

st.caption(
    "🌸 ATM strikes highlighted | 🟢 Positive | 🔴 Negative | "
    "Live snapshot (HH:MM) • Auto-refresh 30s"
)
