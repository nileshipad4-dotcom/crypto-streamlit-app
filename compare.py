import streamlit as st
import pandas as pd
import requests

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

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
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "20-12-2025"

UNDERLYING = st.sidebar.selectbox("Underlying", ["BTC", "ETH"])
CSV_PATH = f"data/{UNDERLYING}.csv"

# CSV column indices (0-based)
STRIKE_COL_IDX = 6
VALUE_COL_IDX = 19
TIMESTAMP_COL_IDX = 14
CALL_GAMMA_COL_IDX = 20
PUT_GAMMA_COL_IDX = 21

GAMMA_FACTOR = 100_000_000

# -------------------------------------------------
# LIVE PRICE
# -------------------------------------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        sym = f"{symbol}USD"
        return float(next(x for x in r if x["symbol"] == sym)["mark_price"])
    except:
        return None

price = get_delta_price(UNDERLYING)

st.sidebar.metric(
    f"{UNDERLYING} Price (Delta)",
    f"{int(price):,}" if price else "Error"
)

# -------------------------------------------------
# LOAD CSV DATA
# -------------------------------------------------
df_raw = pd.read_csv(CSV_PATH)

df = pd.DataFrame({
    "strike_price": pd.to_numeric(df_raw.iloc[:, STRIKE_COL_IDX], errors="coerce"),
    "value": pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),
    "call_gamma": pd.to_numeric(df_raw.iloc[:, CALL_GAMMA_COL_IDX], errors="coerce"),
    "put_gamma": pd.to_numeric(df_raw.iloc[:, PUT_GAMMA_COL_IDX], errors="coerce"),
    "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]
}).dropna(subset=["strike_price", "timestamp"])

# -------------------------------------------------
# TIME SELECTION
# -------------------------------------------------
timestamps = sorted(df["timestamp"].unique(), reverse=True)

if len(timestamps) < 2:
    st.error("Not enough timestamps in CSV")
    st.stop()

t1 = st.selectbox("Select Time 1 (Latest)", timestamps, index=0)
t2 = st.selectbox("Select Time 2 (Previous)", timestamps, index=1)

if t1 == t2:
    st.warning("Select two different timestamps")
    st.stop()

# -------------------------------------------------
# VALUE COMPARISON
# -------------------------------------------------
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
merged["Change"] = merged[t1] - merged[t2]

# -------------------------------------------------
# GAMMA AT TIME-1
# -------------------------------------------------
gamma_t1 = (
    df[df["timestamp"] == t1]
    .groupby("strike_price", as_index=False)
    .agg({
        "call_gamma": "sum",
        "put_gamma": "sum"
    })
    .rename(columns={
        "call_gamma": "call_gamma_t1",
        "put_gamma": "put_gamma_t1"
    })
)

# -------------------------------------------------
# FETCH LIVE OPTION CHAIN
# -------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_chain():
    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={UNDERLYING}"
        f"&expiry_date={EXPIRY}"
    )
    return pd.json_normalize(
        requests.get(url, timeout=20).json()["result"]
    )

df_live = fetch_live_chain()[[
    "strike_price",
    "contract_type",
    "mark_price",
    "oi_contracts",
    "greeks.gamma"
]].rename(columns={"greeks.gamma": "gamma"})

for c in ["strike_price", "gamma"]:
    df_live[c] = pd.to_numeric(df_live[c], errors="coerce")

calls = (
    df_live[df_live["contract_type"] == "call_options"]
    .groupby("strike_price", as_index=False)["gamma"]
    .sum()
    .rename(columns={"gamma": "call_gamma_live"})
)

puts = (
    df_live[df_live["contract_type"] == "put_options"]
    .groupby("strike_price", as_index=False)["gamma"]
    .sum()
    .rename(columns={"gamma": "put_gamma_live"})
)

live_gamma = pd.merge(calls, puts, on="strike_price", how="outer")

# -------------------------------------------------
# COMPUTE LIVE MAX PAIN
# -------------------------------------------------
df_live_mp = fetch_live_chain()[[
    "strike_price", "contract_type", "mark_price", "oi_contracts"
]]

for c in ["strike_price", "mark_price", "oi_contracts"]:
    df_live_mp[c] = pd.to_numeric(df_live_mp[c], errors="coerce")

calls_mp = df_live_mp[df_live_mp["contract_type"] == "call_options"]
puts_mp  = df_live_mp[df_live_mp["contract_type"] == "put_options"]

live_mp = pd.merge(
    calls_mp.rename(columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}),
    puts_mp.rename(columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}),
    on="strike_price",
    how="outer"
)

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

    df["Current"] = mp
    return df[["strike_price", "Current"]]

live_mp = compute_max_pain(live_mp)

# -------------------------------------------------
# FINAL MERGE
# -------------------------------------------------
final = (
    merged
    .merge(live_mp, on="strike_price", how="left")
    .merge(gamma_t1, on="strike_price", how="left")
    .merge(live_gamma, on="strike_price", how="left")
)

final["Current − Time1"] = final["Current"] - final[t1]

final["Call Γ Δ"] = (final["call_gamma_live"] - final["call_gamma_t1"]) * GAMMA_FACTOR
final["Put Γ Δ"]  = (final["put_gamma_live"]  - final["put_gamma_t1"])  * GAMMA_FACTOR

final = final[[
    "strike_price",
    "Current",
    "Current − Time1",
    "Call Γ Δ",
    "Put Γ Δ",
    t1,
    t2,
    "Change"
]].sort_values("strike_price")

for c in final.columns:
    final[c] = final[c].round(0).astype("Int64")

# -------------------------------------------------
# ATM RANGE
# -------------------------------------------------
lower_strike = upper_strike = None
if price:
    strikes = final["strike_price"].astype(float).tolist()
    lower = [s for s in strikes if s <= price]
    upper = [s for s in strikes if s >= price]
    if lower: lower_strike = max(lower)
    if upper: upper_strike = min(upper)

# -------------------------------------------------
# STYLING
# -------------------------------------------------
def color_values(v):
    if pd.isna(v): return ""
    if v > 0: return "background-color: lightgreen"
    if v < 0: return "background-color: lightcoral"
    return ""

def highlight_price_range(row):
    styles = [""] * len(row)
    if row["strike_price"] in (lower_strike, upper_strike):
        styles[0] = styles[1] = "background-color: #ffd6e7"
    return styles

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")

st.dataframe(
    final.style
        .applymap(color_values, subset=["Change", "Current − Time1", "Call Γ Δ", "Put Γ Δ"])
        .apply(highlight_price_range, axis=1),
    use_container_width=True
)

st.caption(
    "🌸 ATM | 🟢 Positive | 🔴 Negative | "
    "Gamma Δ scaled × 100,000,000 | Live refresh 30s"
)
