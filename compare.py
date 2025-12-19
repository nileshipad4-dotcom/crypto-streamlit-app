import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

# -------------------------------------------------
# AUTO REFRESH (60s)
# -------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60 * 1000, key="refresh")
except ImportError:
    pass

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
EXPIRY = "20-12-2025"

UNDERLYING = st.sidebar.selectbox("Underlying", ["BTC", "ETH"])
CSV_PATH = f"data/{UNDERLYING}.csv"

STRIKE_COL_IDX = 6
TIMESTAMP_COL_IDX = 14
VALUE_COL_IDX = 19

CALL_OI_COL_IDX = 1
PUT_OI_COL_IDX = 11
CALL_VOL_COL_IDX = 2
PUT_VOL_COL_IDX = 10

CALL_GAMMA_COL_IDX = 3
CALL_DELTA_COL_IDX = 4
CALL_VEGA_COL_IDX = 5
PUT_GAMMA_COL_IDX = 7
PUT_DELTA_COL_IDX = 8
PUT_VEGA_COL_IDX = 9

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
st.sidebar.metric(f"{UNDERLYING} Price", f"{int(price):,}" if price else "Error")

# -------------------------------------------------
# LOAD CSV DATA
# -------------------------------------------------
df_raw = pd.read_csv(CSV_PATH)

df = pd.DataFrame({
    "strike_price": pd.to_numeric(df_raw.iloc[:, STRIKE_COL_IDX], errors="coerce"),
    "value": pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),
    "call_oi": pd.to_numeric(df_raw.iloc[:, CALL_OI_COL_IDX], errors="coerce"),
    "put_oi": pd.to_numeric(df_raw.iloc[:, PUT_OI_COL_IDX], errors="coerce"),
    "call_vol": pd.to_numeric(df_raw.iloc[:, CALL_VOL_COL_IDX], errors="coerce"),
    "put_vol": pd.to_numeric(df_raw.iloc[:, PUT_VOL_COL_IDX], errors="coerce"),
    "call_gamma": pd.to_numeric(df_raw.iloc[:, CALL_GAMMA_COL_IDX], errors="coerce"),
    "call_delta": pd.to_numeric(df_raw.iloc[:, CALL_DELTA_COL_IDX], errors="coerce"),
    "call_vega": pd.to_numeric(df_raw.iloc[:, CALL_VEGA_COL_IDX], errors="coerce"),
    "put_gamma": pd.to_numeric(df_raw.iloc[:, PUT_GAMMA_COL_IDX], errors="coerce"),
    "put_delta": pd.to_numeric(df_raw.iloc[:, PUT_DELTA_COL_IDX], errors="coerce"),
    "put_vega": pd.to_numeric(df_raw.iloc[:, PUT_VEGA_COL_IDX], errors="coerce"),
    "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5],
}).dropna(subset=["strike_price", "timestamp"])

# -------------------------------------------------
# TIME SELECTION
# -------------------------------------------------
timestamps = sorted(df["timestamp"].unique(), reverse=True)
t1 = st.selectbox("Select Time 1 (Latest)", timestamps, index=0)
t2 = st.selectbox("Select Time 2 (Previous)", timestamps, index=1)

# -------------------------------------------------
# PCR FROM CSV SNAPSHOTS
# -------------------------------------------------
def compute_pcr(d):
    return (
        d["put_oi"].sum() / d["call_oi"].sum() if d["call_oi"].sum() else None,
        d["put_vol"].sum() / d["call_vol"].sum() if d["call_vol"].sum() else None,
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

call_oi = df_live.loc[df_live["contract_type"] == "call_options", "oi_contracts"].sum()
put_oi = df_live.loc[df_live["contract_type"] == "put_options", "oi_contracts"].sum()
call_vol = df_live.loc[df_live["contract_type"] == "call_options", "volume"].sum()
put_vol = df_live.loc[df_live["contract_type"] == "put_options", "volume"].sum()

pcr_live_oi = put_oi / call_oi if call_oi else None
pcr_live_vol = put_vol / call_vol if call_vol else None

# -------------------------------------------------
# PCR TABLE
# -------------------------------------------------
pcr_table = pd.DataFrame(
    {
        "Current": [pcr_live_oi, pcr_live_vol],
        t1: [pcr_t1_oi, pcr_t1_vol],
        t2: [pcr_t2_oi, pcr_t2_vol],
    },
    index=["PCR OI", "PCR Volume"],
).applymap(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")

st.subheader(f"{UNDERLYING} PCR Snapshot")
st.dataframe(pcr_table, use_container_width=True)

# -------------------------------------------------
# HISTORICAL MAX PAIN
# -------------------------------------------------
df_t1 = df[df["timestamp"] == t1].groupby("strike_price", as_index=False)["value"].sum().rename(columns={"value": t1})
df_t2 = df[df["timestamp"] == t2].groupby("strike_price", as_index=False)["value"].sum().rename(columns={"value": t2})

merged = pd.merge(df_t1, df_t2, on="strike_price", how="outer")
merged["Change"] = merged[t1] - merged[t2]

# -------------------------------------------------
# LIVE MAX PAIN
# -------------------------------------------------
df_mp = df_live[["strike_price", "contract_type", "mark_price", "oi_contracts"]].copy()
for c in ["strike_price", "mark_price", "oi_contracts"]:
    df_mp[c] = pd.to_numeric(df_mp[c], errors="coerce")

calls_mp = df_mp[df_mp["contract_type"] == "call_options"]
puts_mp = df_mp[df_mp["contract_type"] == "put_options"]

live_mp = pd.merge(
    calls_mp.rename(columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}),
    puts_mp.rename(columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}),
    on="strike_price",
    how="outer",
).sort_values("strike_price")

def compute_max_pain(df):
    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    df["Current"] = [
        round(
            (
                -sum(A[i:] * B[i:])
                + G[i] * sum(B[:i])
                - sum(G[:i] * B[:i])
                - sum(M[:i] * L[:i])
                + sum(G[i:] * L[i:])
                - G[i] * sum(L[i:])
            )
            / 10000
        )
        for i in range(len(df))
    ]
    return df[["strike_price", "Current"]]

live_mp = compute_max_pain(live_mp)

# -------------------------------------------------
# FINAL MERGE
# -------------------------------------------------
final = (
    merged.merge(live_mp, on="strike_price", how="left")
    .merge(
        df[df["timestamp"] == t1]
        .groupby("strike_price", as_index=False)
        .agg(
            call_gamma_t1=("call_gamma", "sum"),
            call_delta_t1=("call_delta", "sum"),
            call_vega_t1=("call_vega", "sum"),
            put_gamma_t1=("put_gamma", "sum"),
            put_delta_t1=("put_delta", "sum"),
            put_vega_t1=("put_vega", "sum"),
        ),
        on="strike_price",
        how="left",
    )
)

# -------------------------------------------------
# RENAME + ORDER
# -------------------------------------------------
now_ts = get_ist_time()

final = final.rename(columns={
    "Current": f"MP ({now_ts})",
    t1: f"MP ({t1})",
    t2: f"MP ({t2})",
    "Change": "△ MP 2",
})

final = final.round(0).astype("Int64")

# -------------------------------------------------
# ATM HIGHLIGHT
# -------------------------------------------------
atm_low = atm_high = None
if price:
    strikes = final["strike_price"].astype(float)
    atm_low = strikes[strikes <= price].max()
    atm_high = strikes[strikes >= price].min()

def highlight_atm(row):
    if row["strike_price"] in (atm_low, atm_high):
        return ["background-color: #000435; color: white"] * len(row)
    return [""] * len(row)

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")

st.dataframe(
    final.style.apply(highlight_atm, axis=1),
    use_container_width=True,
    height=700,
)

st.caption("🟦 ATM band | MP = Max Pain | △ = Delta | PCR shown above")
