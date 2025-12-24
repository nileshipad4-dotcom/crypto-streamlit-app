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
EXPIRY = "25-12-2025"

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
st.sidebar.metric(
    f"{UNDERLYING} Price",
    f"{int(price):,}" if price else "Error"
)

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
def rotated_time_sort(times, pivot="17:30"):
    pivot_h, pivot_m = map(int, pivot.split(":"))
    pivot_minutes = pivot_h * 60 + pivot_m

    def key(t):
        h, m = map(int, t.split(":"))
        total = h * 60 + m
        return (total - pivot_minutes) % (24 * 60)

    return sorted(times, key=key, reverse=True)

timestamps = rotated_time_sort(df["timestamp"].unique(), pivot="17:30")

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
    return pd.json_normalize(
        requests.get(url, timeout=20).json()["result"]
    )

df_live = fetch_live_chain()

# -------------------------------------------------
# LIVE PCR
# -------------------------------------------------
df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")
df_live["volume"] = pd.to_numeric(df_live["volume"], errors="coerce")

pcr_live_oi = (
    df_live[df_live["contract_type"] == "put_options"]["oi_contracts"].sum()
    / df_live[df_live["contract_type"] == "call_options"]["oi_contracts"].sum()
)

pcr_live_vol = (
    df_live[df_live["contract_type"] == "put_options"]["volume"].sum()
    / df_live[df_live["contract_type"] == "call_options"]["volume"].sum()
)

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
# LIVE MAX PAIN
# -------------------------------------------------
df_mp = df_live[[
    "strike_price",
    "contract_type",
    "mark_price",
    "oi_contracts"
]].copy()

for c in ["strike_price", "mark_price", "oi_contracts"]:
    df_mp[c] = pd.to_numeric(df_mp[c], errors="coerce")

calls_mp = df_mp[df_mp["contract_type"] == "call_options"]
puts_mp = df_mp[df_mp["contract_type"] == "put_options"]

live_mp = (
    pd.merge(
        calls_mp.rename(columns={
            "mark_price": "call_mark",
            "oi_contracts": "call_oi"
        }),
        puts_mp.rename(columns={
            "mark_price": "put_mark",
            "oi_contracts": "put_oi"
        }),
        on="strike_price",
        how="outer",
    )
    .sort_values("strike_price")
)

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
            ) / 10000
        )
        for i in range(len(df))
    ]

    return df[["strike_price", "Current"]]

live_mp = compute_max_pain(live_mp)

# -------------------------------------------------
# GREEKS (TIME-1 vs LIVE)
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

df_g = df_live[[
    "strike_price",
    "contract_type",
    "greeks.gamma",
    "greeks.delta",
    "greeks.vega"
]].copy()

df_g.columns = ["strike_price", "contract_type", "gamma", "delta", "vega"]

for c in ["strike_price", "gamma", "delta", "vega"]:
    df_g[c] = pd.to_numeric(df_g[c], errors="coerce")

calls = df_g[df_g["contract_type"] == "call_options"]
puts = df_g[df_g["contract_type"] == "put_options"]

live_greeks = (
    calls.groupby("strike_price", as_index=False)
    .agg(
        call_gamma_live=("gamma", "sum"),
        call_delta_live=("delta", "sum"),
        call_vega_live=("vega", "sum"),
    )
    .merge(
        puts.groupby("strike_price", as_index=False)
        .agg(
            put_gamma_live=("gamma", "sum"),
            put_delta_live=("delta", "sum"),
            put_vega_live=("vega", "sum"),
        ),
        on="strike_price",
        how="outer",
    )
)

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
final["ΔΔ MP 1"] = -1 * (
    final["Current − Time1"].shift(-1) - final["Current − Time1"]
)

final["Call Gamma △"] = (
    final["call_gamma_live"] - final["call_gamma_t1"]
) * FACTOR / 100

final["Put Gamma △"] = (
    final["put_gamma_live"] - final["put_gamma_t1"]
) * FACTOR / 100

final["Call Delta △"] = (
    final["call_delta_live"] - final["call_delta_t1"]
) * FACTOR / 100000

final["Put Delta △"] = (
    final["put_delta_live"] - final["put_delta_t1"]
) * FACTOR / -100000

final["Call Vega △"] = (
    final["call_vega_live"] - final["call_vega_t1"]
) * FACTOR / 1000000

final["Put Vega △"] = (
    final["put_vega_live"] - final["put_vega_t1"]
) * FACTOR / 1000000

# -------------------------------------------------
# RENAME + ORDER
# -------------------------------------------------
now_ts = get_ist_time()

final = final.rename(columns={
    "Current": f"MP ({now_ts})",
    t1: f"MP ({t1})",
    t2: f"MP ({t2})",
    "Current − Time1": "△ MP 1",
    "Change": "△ MP 2",
})

final = final[
    [
        "strike_price",
        f"MP ({now_ts})",
        f"MP ({t1})",
        "△ MP 1",
        "ΔΔ MP 1",
        f"MP ({t2})",
        "△ MP 2",
        "Call Gamma △",
        "Put Gamma △",
        "Call Delta △",
        "Put Delta △",
        "Call Vega △",
        "Put Vega △",
    ]
].round(0).astype("Int64")

# -------------------------------------------------
# ATM HIGHLIGHT LOGIC
# -------------------------------------------------
mp_cur = f"MP ({now_ts})"

atm_low = atm_high = None
if price:
    strikes = final["strike_price"].astype(float).tolist()
    below = [s for s in strikes if s <= price]
    above = [s for s in strikes if s >= price]

    if below:
        atm_low = max(below)
    if above:
        atm_high = min(above)

def highlight_atm(row):
    if row["strike_price"] in (atm_low, atm_high):
        return ["background-color: #000435"] * len(row)
    return [""] * len(row)

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
st.subheader(f"{UNDERLYING} Comparison — {t1} vs {t2}")

styled = final.style.apply(highlight_atm, axis=1)

st.dataframe(
    styled,
    use_container_width=True,
    height=700,
    column_config={
        "strike_price": st.column_config.NumberColumn("Strike", pinned=True),
        mp_cur: st.column_config.NumberColumn(mp_cur, pinned=True),
        f"MP ({t1})": st.column_config.NumberColumn(f"MP ({t1})", pinned=True),
        "△ MP 1": st.column_config.NumberColumn("△ MP 1", pinned=True),
    },
)

st.caption(
    "🟡 ATM band | MP = Max Pain | △ = Live − Time1 | PCR shown above"
)


