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
# PCR CALCULATION (CSV SNAPSHOTS)
# -------------------------------------------------
def compute_pcr(snapshot_df):
    call_oi = snapshot_df["call_oi"].sum()
    put_oi  = snapshot_df["put_oi"].sum()
    call_v  = snapshot_df["call_vol"].sum()
    put_v   = snapshot_df["put_vol"].sum()

    return (
        (put_oi / call_oi) if call_oi else None,
        (put_v  / call_v)  if call_v  else None
    )

pcr_t1_oi, pcr_t1_vol = compute_pcr(df[df["timestamp"] == t1])
pcr_t2_oi, pcr_t2_vol = compute_pcr(df[df["timestamp"] == t2])

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
# LIVE PCR
# -------------------------------------------------
df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")
df_live["volume"] = pd.to_numeric(df_live["volume"], errors="coerce")

call_oi_live = df_live.loc[df_live["contract_type"]=="call_options","oi_contracts"].sum()
put_oi_live  = df_live.loc[df_live["contract_type"]=="put_options","oi_contracts"].sum()
call_vol_live = df_live.loc[df_live["contract_type"]=="call_options","volume"].sum()
put_vol_live  = df_live.loc[df_live["contract_type"]=="put_options","volume"].sum()

pcr_live_oi  = put_oi_live / call_oi_live if call_oi_live else None
pcr_live_vol = put_vol_live / call_vol_live if call_vol_live else None

# -------------------------------------------------
# DISPLAY PCR SUMMARY (ABOVE MAIN TABLE)
# -------------------------------------------------
pcr_df = pd.DataFrame({
    "Snapshot": ["Current", t1, t2],
    "PCR OI": [
        f"{pcr_live_oi:.3f}" if pcr_live_oi else "NA",
        f"{pcr_t1_oi:.3f}"   if pcr_t1_oi   else "NA",
        f"{pcr_t2_oi:.3f}"   if pcr_t2_oi   else "NA",
    ],
    "PCR Volume": [
        f"{pcr_live_vol:.3f}" if pcr_live_vol else "NA",
        f"{pcr_t1_vol:.3f}"   if pcr_t1_vol   else "NA",
        f"{pcr_t2_vol:.3f}"   if pcr_t2_vol   else "NA",
    ]
})

st.subheader(f"{UNDERLYING} PCR Snapshot")
st.dataframe(pcr_df, use_container_width=True, hide_index=True)

# -------------------------------------------------
# (REST OF YOUR CODE: max pain, greeks, final table)
# -------------------------------------------------
# ⬇️ KEEP EVERYTHING ELSE EXACTLY AS YOU ALREADY HAVE IT
# ⬇️ NO CHANGES REQUIRED BELOW THIS POINT
