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

STRIKE_COL_IDX    = 6
TIMESTAMP_COL_IDX = 14
VALUE_COL_IDX     = 19

CALL_OI_COL_IDX   = 1
PUT_OI_COL_IDX    = 11
CALL_VOL_COL_IDX  = 2
PUT_VOL_COL_IDX   = 10

CALL_GAMMA_COL_IDX = 3
CALL_DELTA_COL_IDX = 4
CALL_VEGA_COL_IDX  = 5
PUT_GAMMA_COL_IDX  = 7
PUT_DELTA_COL_IDX  = 8
PUT_VEGA_COL_IDX   = 9

FACTOR = 100_000_000

# -------------------------------------------------
# LIVE PRICE (keep cached, OK)
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
    "call_oi":      pd.to_numeric(df_raw.iloc[:, CALL_OI_COL_IDX], errors="coerce"),
    "put_oi":       pd.to_numeric(df_raw.iloc[:, PUT_OI_COL_IDX], errors="coerce"),
    "call_vol":     pd.to_numeric(df_raw.iloc[:, CALL_VOL_COL_IDX], errors="coerce"),
    "put_vol":      pd.to_numeric(df_raw.iloc[:, PUT_VOL_COL_IDX], errors="coerce"),
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
t1 = st.selectbox("Select Time 1 (Latest)", timestamps, index=0)
t2 = st.selectbox("Select Time 2 (Previous)", timestamps, index=1)

# -------------------------------------------------
# FETCH LIVE OPTION CHAIN (❗ NO CACHE)
# -------------------------------------------------
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
# LIVE MAX PAIN
# -------------------------------------------------
df_mp = df_live[["strike_price","contract_type","mark_price","oi_contracts"]].copy()
for c in df_mp.columns:
    df_mp[c] = pd.to_numeric(df_mp[c], errors="coerce")

calls_mp = df_mp[df_mp["contract_type"]=="call_options"]
puts_mp  = df_mp[df_mp["contract_type"]=="put_options"]

live_mp = pd.merge(
    calls_mp.rename(columns={"mark_price":"call_mark","oi_contracts":"call_oi"}),
    puts_mp.rename(columns={"mark_price":"put_mark","oi_contracts":"put_oi"}),
    on="strike_price",
    how="outer"
).sort_values("strike_price").reset_index(drop=True)

def compute_max_pain(df):
    A,B,G,L,M = (
        df["call_mark"].fillna(0).values,
        df["call_oi"].fillna(0).values,
        df["strike_price"].values,
        df["put_oi"].fillna(0).values,
        df["put_mark"].fillna(0).values
    )
    mp=[]
    for i in range(len(df)):
        mp.append(round((
            -sum(A[i:]*B[i:]) +
            G[i]*sum(B[:i]) - sum(G[:i]*B[:i]) -
            sum(M[:i]*L[:i]) +
            sum(G[i:]*L[i:]) - G[i]*sum(L[i:])
        )/10000))
    df["Current"] = mp
    return df[["strike_price","Current"]]

live_mp = compute_max_pain(live_mp)

# -------------------------------------------------
# HISTORICAL COMPARISON
# -------------------------------------------------
df_t1 = df[df["timestamp"]==t1].groupby("strike_price",as_index=False)["value"].sum().rename(columns={"value":t1})
df_t2 = df[df["timestamp"]==t2].groupby("strike_price",as_index=False)["value"].sum().rename(columns={"value":t2})

final = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
final = final.merge(live_mp, on="strike_price", how="left")

final["Current"] = pd.to_numeric(final["Current"], errors="coerce")
final["Current − Time1"] = final["Current"] - final[t1]

final = final.sort_values("strike_price").round(0)

# -------------------------------------------------
# DISPLAY (PINNED)
# -------------------------------------------------
st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")

st.dataframe(
    final,
    use_container_width=True,
    height=700,
    column_config={
        "strike_price": st.column_config.NumberColumn("Strike", pinned=True),
        "Current": st.column_config.NumberColumn("Max Pain (Live)", pinned=True),
    },
)

st.caption("Live data refreshes every 30s | Strike & Max Pain pinned")
