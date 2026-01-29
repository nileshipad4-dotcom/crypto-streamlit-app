import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("📊 Live Max Pain — Strike-wise (Single Expiry)")

# =================================================
# AUTO REFRESH (60s)
# =================================================
if st_autorefresh(interval=60_000, key="auto_refresh"):
    st.cache_data.clear()

# =================================================
# CONSTANTS
# =================================================
API_BASE = "https://api.india.delta.exchange/v2/tickers"
ASSETS = ["BTC", "ETH"]

# =================================================
# HELPERS
# =================================================
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_next_10_expiries():
    """Next 10 calendar expiries with day name"""
    start = get_ist_datetime().date()
    expiries = []

    for i in range(10):
        d = start + timedelta(days=i)
        expiries.append({
            "label": f"{d.strftime('%d-%m-%Y')} ({d.strftime('%a')})",
            "value": d.strftime("%d-%m-%Y")
        })

    return expiries

def fetch_option_chain(asset, expiry):
    r = requests.get(
        f"{API_BASE}?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={asset}"
        f"&expiry_date={expiry}",
        timeout=20
    ).json()["result"]

    if not r:
        return pd.DataFrame()

    df = pd.json_normalize(r)

    for c in ["strike_price", "mark_price", "oi_contracts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["strike_price", "oi_contracts"])

# =================================================
# YOUR EXACT MAX PAIN FORMULA (UNCHANGED)
# =================================================
def compute_max_pain(df):

    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    n = len(df)
    U = []

    for i in range(n):
        q = -sum(A[i:] * B[i:])
        r = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        s = -sum(M[:i] * L[:i])
        t = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        U.append(round((q + r + s + t) / 10000))

    df["max_pain"] = U
    return df

# =================================================
# CONTROLS
# =================================================
c1, c2 = st.columns([1, 2])

with c1:
    asset = st.selectbox("Asset", ASSETS)

with c2:
    expiry_options = get_next_10_expiries()
    expiry_label_map = {e["label"]: e["value"] for e in expiry_options}

    selected_label = st.selectbox(
        "Expiry",
        list(expiry_label_map.keys())
    )

selected_expiry = expiry_label_map[selected_label]

# =================================================
# MAIN LOGIC
# =================================================
raw = fetch_option_chain(asset, selected_expiry)

if raw.empty:
    st.warning("No option data available for this expiry.")
    st.stop()

calls = raw[raw["contract_type"] == "call_options"][
    ["strike_price", "mark_price", "oi_contracts"]
].rename(columns={
    "mark_price": "call_mark",
    "oi_contracts": "call_oi"
})

puts = raw[raw["contract_type"] == "put_options"][
    ["strike_price", "mark_price", "oi_contracts"]
].rename(columns={
    "mark_price": "put_mark",
    "oi_contracts": "put_oi"
})

mp_df = (
    pd.merge(calls, puts, on="strike_price", how="outer")
    .fillna(0)
    .sort_values("strike_price")
    .reset_index(drop=True)
)

mp_df = compute_max_pain(mp_df)

final = mp_df[["strike_price", "max_pain"]]

# =================================================
# DISPLAY
# =================================================
min_pain = final["max_pain"].min()

def highlight(row):
    if row["max_pain"] == min_pain:
        return ["background-color:#8B0000;color:white"] * len(row)
    return [""] * len(row)

st.subheader(f"{asset} — {selected_label}")
st.dataframe(
    final.style.apply(highlight, axis=1).format({"max_pain": "{:,.0f}"}),
    use_container_width=True
)
