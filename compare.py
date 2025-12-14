import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Max Pain (BTC)")

# -------------------------------------------------
# AUTO REFRESH (30s)
# -------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30 * 1000, key="refresh")
except:
    pass

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BTC_PATH = "data/BTC.csv"

STRIKE_COL_IDX = 6      # G:G
VALUE_COL_IDX = 19     # Column to compare
TIMESTAMP_COL_IDX = 14 # O:O

API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "15-12-2025"

# -------------------------------------------------
# LIVE PRICE (BTC & ETH)
# -------------------------------------------------
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
merged["change"] = merged["value_time_1"] - merged["value_time_2"]

# -------------------------------------------------
# FETCH LIVE OPTION CHAIN
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

# -------------------------------------------------
# COMPUTE LIVE MAX PAIN
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
# FINAL TABLE
# -------------------------------------------------
final = pd.merge(
    merged,
    df_live_mp,
    on="strike_price",
    how="left"
).sort_values("strike_price")

final["mp_minus_time1"] = final["max_pain"] - final["value_time_1"]

final = final[
    ["strike_price", "max_pain", "mp_minus_time1", "value_time_1", "value_time_2", "change"]
]

for col in final.columns:
    final[col] = final[col].round(0).astype("Int64")

def color_change(v):
    if pd.isna(v): return ""
    if v > 0: return "background-color: lightgreen"
    if v < 0: return "background-color: lightcoral"
    return ""

st.subheader(f"Comparison {t1} → {t2} + Live Max Pain")

st.dataframe(
    final.style.applymap(color_change, subset=["change", "mp_minus_time1"]),
    use_container_width=True
)

# -------------------------------------------------
# 🔥 BTC 3-MIN CANDLESTICK CHART + EMA 34
# -------------------------------------------------
st.markdown("---")
st.subheader("📉 BTC 3-Minute Candlestick Chart + EMA-34")

@st.cache_data(ttl=30)
def fetch_btc_3m():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "3m", "limit": 200}
    data = requests.get(url, params=params, timeout=10).json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbv","tqv","ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c])

    df["ema_high"] = df["high"].ewm(span=34, adjust=False).mean()
    df["ema_low"]  = df["low"].ewm(span=34, adjust=False).mean()
    return df

ohlc = fetch_btc_3m()

fig = go.Figure()

fig.add_candlestick(
    x=ohlc["open_time"],
    open=ohlc["open"],
    high=ohlc["high"],
    low=ohlc["low"],
    close=ohlc["close"],
    increasing_line_color="green",
    decreasing_line_color="red",
    name="BTC 3m"
)

fig.add_scatter(
    x=ohlc["open_time"],
    y=ohlc["ema_high"],
    line=dict(color="orange", width=2),
    name="EMA 34 High"
)

fig.add_scatter(
    x=ohlc["open_time"],
    y=ohlc["ema_low"],
    line=dict(color="blue", width=2),
    name="EMA 34 Low"
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.caption("🟢 Bullish Candle | 🔴 Bearish Candle | EMA-34 High & Low • Auto-refresh 30s")

