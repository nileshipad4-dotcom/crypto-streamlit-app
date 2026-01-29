import streamlit as st
import pandas as pd
import requests
import calendar
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(layout="wide")
st.title("📊 Live Max Pain Matrix (Strike × Expiry)")

# =================================================
# AUTO REFRESH
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

def get_expiry_list():
    """Current day + next 5 Fridays"""
    today = get_ist_datetime().date()
    expiries = [today]

    d = today
    while len(expiries) < 6:
        d += timedelta(days=1)
        if d.weekday() == calendar.FRIDAY:
            expiries.append(d)

    return [e.strftime("%d-%m-%Y") for e in expiries]

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
# MAIN
# =================================================
expiries = get_expiry_list()

for asset in ASSETS:
    st.subheader(f"{asset} — Live Max Pain by Strike")

    expiry_tables = []

    for expiry in expiries:
        raw = fetch_option_chain(asset, expiry)
        if raw.empty:
            continue

        # Build strike-wise merged frame
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

        expiry_tables.append(
            mp_df.set_index("strike_price")["max_pain"]
                  .rename(expiry)
        )

    if not expiry_tables:
        st.warning("No live option data available.")
        continue

    final = pd.concat(expiry_tables, axis=1).sort_index()

    # Highlight minimum max pain per expiry (true MP)
    def highlight_min(col):
        min_val = col.min()
        return [
            "background-color:#8B0000;color:white"
            if v == min_val else ""
            for v in col
        ]

    st.dataframe(
        final.style
             .apply(highlight_min, axis=0)
             .format("{:,.0f}"),
        use_container_width=True
    )
