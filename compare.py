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
except ImportError:
    pass

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "20-12-2025"

UNDERLYING = st.sidebar.selectbox("Underlying", ["BTC", "ETH"])
CSV_PATH = f"data/{UNDERLYING}.csv"

# CSV column indices (from collector.py)
STRIKE_COL_IDX    = 6
TIMESTAMP_COL_IDX = 14
VALUE_COL_IDX     = 19   # max_pain snapshot

CALL_GAMMA_COL_IDX = 3
CALL_DELTA_COL_IDX = 4
CALL_VEGA_COL_IDX  = 5

PUT_GAMMA_COL_IDX  = 7
PUT_DELTA_COL_IDX  = 8
PUT_VEGA_COL_IDX   = 9

FACTOR = 100_000_000

# -------------------------------------------------
# LIVE PRICE
# -------------------------------------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        sym = f"{symbol}USD"
        return float(next(x for x in r if x["symbol"] == sym)["mark_price"])
    except Exception:
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
    "value":        pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),

    "call_gamma":   pd.to_numeric(df_raw.iloc[:, CALL_GAMMA_COL_IDX], errors="coerce"),
    "call_delta":   pd.to_numeric(df_raw.iloc[:, CALL_DELTA_COL_IDX], errors="coerce"),
    "call_vega":    pd.to_numeric(df_raw.iloc[:, CALL_VEGA_COL_IDX],  errors="coerce"),

    "put_gamma":    pd.to_numeric(df_raw.iloc[:, PUT_GAMMA_COL_IDX], errors="coerce"),
    "put_delta":    pd.to_numeric(df_raw.iloc[:, PUT_DELTA_COL_IDX], errors="coerce"),
    "put_vega":     pd.to_numeric(df_raw.iloc[:, PUT_VEGA_COL_IDX],  errors="coerce"),

    "timestamp":    df_raw.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]
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

# -------------------------------------------------
# HISTORICAL MAX PAIN COMPARISON
# -------------------------------------------------
df_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False)["value"].sum().rename(columns={"value": t1})
df_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False)["value"].sum().rename(columns={"value": t2})

merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
merged["Change"] = merged[t1] - merged[t2]

# -------------------------------------------------
# GREEKS AT TIME-1
# -------------------------------------------------
greeks_t1 = (
    df[df["timestamp"] == t1]
    .groupby("strike_price", as_index=False)
    .agg(
        call_gamma_t1=("call_gamma", "sum"),
        call_delta_t1=("call_delta", "sum"),
        call_vega_t1=("call_vega", "sum"),

        put_gamma_t1=("put_gamma", "sum"),
        put_delta_t1=("put_delta", "sum"),
        put_vega_t1=("put_vega", "sum"),
    )
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
    return pd.json_normalize(requests.get(url, timeout=20).json()["result"])

df_live = fetch_live_chain()

# -------------------------------------------------
# LIVE GREEKS
# -------------------------------------------------
df_greeks = df_live[[
    "strike_price", "contract_type",
    "greeks.gamma", "greeks.delta", "greeks.vega"
]].rename(columns={
    "greeks.gamma": "gamma",
    "greeks.delta": "delta",
    "greeks.vega": "vega"
})

for c in ["strike_price", "gamma", "delta", "vega"]:
    df_greeks[c] = pd.to_numeric(df_greeks[c], errors="coerce")

calls = df_greeks[df_greeks["contract_type"] == "call_options"]
puts  = df_greeks[df_greeks["contract_type"] == "put_options"]

live_greeks = (
    calls.groupby("strike_price", as_index=False)
    .agg(
        call_gamma_live=("gamma","sum"),
        call_delta_live=("delta","sum"),
        call_vega_live=("vega","sum")
    )
    .merge(
        puts.groupby("strike_price", as_index=False)
        .agg(
            put_gamma_live=("gamma","sum"),
            put_delta_live=("delta","sum"),
            put_vega_live=("vega","sum")
        ),
        on="strike_price",
        how="outer"
    )
)

# -------------------------------------------------
# LIVE MAX PAIN (FIXED)
# -------------------------------------------------
df_mp = df_live[["strike_price","contract_type","mark_price","oi_contracts"]]

for c in ["strike_price","mark_price","oi_contracts"]:
    df_mp[c] = pd.to_numeric(df_mp[c], errors="coerce")

calls_mp = df_mp[df_mp["contract_type"] == "call_options"]
puts_mp  = df_mp[df_mp["contract_type"] == "put_options"]

live_mp = pd.merge(
    calls_mp.rename(columns={"mark_price":"call_mark","oi_contracts":"call_oi"}),
    puts_mp.rename(columns={"mark_price":"put_mark","oi_contracts":"put_oi"}),
    on="strike_price",
    how="outer"
)

# 🔴 CRITICAL FIX
live_mp = live_mp.sort_values("strike_price").reset_index(drop=True)

def compute_max_pain(df):
    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    mp = []
    for i in range(len(df)):
        mp.append(round((
            -sum(A[i:] * B[i:]) +
            G[i] * sum(B[:i]) - sum(G[:i] * B[:i]) -
            sum(M[:i] * L[:i]) +
            sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        ) / 10000))

    df["Current"] = mp
    return df[["strike_price", "Current"]]

live_mp = compute_max_pain(live_mp)

# -------------------------------------------------
# FINAL MERGE
# -------------------------------------------------
final = (
    merged
    .merge(live_mp, on="strike_price", how="left")
    .merge(greeks_t1, on="strike_price", how="left")
    .merge(live_greeks, on="strike_price", how="left")
)

final["Current − Time1"] = final["Current"] - final[t1]

# ---- GREEK DELTAS (LAST COLUMNS) ----
final["Call Gamma △"] = (final["call_gamma_live"] - final["call_gamma_t1"]) * FACTOR / 10
final["Put Gamma △"]  = (final["put_gamma_live"]  - final["put_gamma_t1"])  * FACTOR / 10

final["Call Delta △"] = (final["call_delta_live"] - final["call_delta_t1"]) * FACTOR / 100000
final["Put Delta △"]  = (final["put_delta_live"]  - final["put_delta_t1"])  * FACTOR / 100000

final["Call Vega △"]  = (final["call_vega_live"]  - final["call_vega_t1"])  * FACTOR / 1000000
final["Put Vega △"]   = (final["put_vega_live"]   - final["put_vega_t1"])   * FACTOR / 1000000

# -------------------------------------------------
# FINAL ORDER
# -------------------------------------------------
final = final[[
    "strike_price","Current","Current − Time1",t1,t2,"Change",
    "Call Gamma △","Put Gamma △",
    "Call Delta △","Put Delta △",
    "Call Vega △","Put Vega △"
]].sort_values("strike_price")

final = final.round(0).astype("Int64")

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")

st.dataframe(final, use_container_width=True)

st.caption(
    "△ = Live − Time1 | Scaled × 100,000,000 | "
    "Live max pain sorted correctly"
)

