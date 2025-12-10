# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from typing import List
import io

# -------------------
# CONFIG
# -------------------
EXPIRIES = ["11-12-2025"]
UNDERLYINGS = ["BTC", "ETH"]
REFRESH_SECONDS = 30
API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}

st.set_page_config(
    page_title="Crypto Option Chain Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# LIVE PRICE FETCHERS
# -------------------
@st.cache_data(ttl=10)
def get_btc_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        r = requests.get(url, timeout=10)
        return r.json().get("bitcoin", {}).get("usd", None)
    except:
        return None

@st.cache_data(ttl=10)
def get_eth_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
        r = requests.get(url, timeout=10)
        return r.json().get("ethereum", {}).get("usd", None)
    except:
        return None


# -------------------
# HELPERS
# -------------------
@st.cache_data(ttl=60)
def fetch_tickers(underlying: str, expiry: str) -> pd.DataFrame:
    params = {
        "contract_types": "call_options,put_options",
        "underlying_asset_symbols": underlying,
        "expiry_date": expiry
    }
    url = f"{API_BASE}?contract_types={params['contract_types']}&underlying_asset_symbols={params['underlying_asset_symbols']}&expiry_date={params['expiry_date']}"

    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    raw = resp.json().get("result", [])
    if not raw:
        return pd.DataFrame()
    return pd.json_normalize(raw)

def safe_to_numeric(val):
    try:
        return pd.to_numeric(val)
    except:
        return val

def format_option_chain(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    needed_cols = [
        "strike_price", "contract_type", "mark_price", "oi_contracts",
        "volume", "greeks.gamma", "greeks.delta", "greeks.vega"
    ]
    for col in needed_cols:
        if col not in df_raw.columns:
            df_raw[col] = None

    df = df_raw[needed_cols].copy()

    df.rename(columns={
        "greeks.gamma": "gamma",
        "greeks.delta": "delta",
        "greeks.vega": "vega",
    }, inplace=True)

    for col in ["strike_price", "mark_price", "oi_contracts", "volume", "gamma", "delta", "vega"]:
        df[col] = df[col].apply(safe_to_numeric)

    calls = df[df["contract_type"] == "call_options"].copy()
    puts = df[df["contract_type"] == "put_options"].copy()

    calls = calls.rename(columns={
        "mark_price": "call_mark",
        "oi_contracts": "call_oi",
        "volume": "call_volume",
        "gamma": "call_gamma",
        "delta": "call_delta",
        "vega": "call_vega"
    }).drop(columns=["contract_type"])

    puts = puts.rename(columns={
        "mark_price": "put_mark",
        "oi_contracts": "put_oi",
        "volume": "put_volume",
        "gamma": "put_gamma",
        "delta": "put_delta",
        "vega": "put_vega"
    }).drop(columns=["contract_type"])

    merged = pd.merge(calls, puts, on="strike_price", how="outer")

    cols = [
        "call_mark", "call_oi", "call_volume",
        "call_gamma", "call_delta", "call_vega",
        "strike_price",
        "put_gamma", "put_delta", "put_vega",
        "put_volume", "put_oi", "put_mark"
    ]

    for c in cols:
        if c not in merged.columns:
            merged[c] = None

    return merged[cols].sort_values("strike_price").reset_index(drop=True)

def fetch_chain_for_underlying(underlying: str, expiries: List[str]) -> pd.DataFrame:
    frames = []
    for e in expiries:
        raw = fetch_tickers(underlying, e)
        if raw.empty:
            continue
        formatted = format_option_chain(raw)
        formatted["expiry"] = e
        frames.append(formatted)
        frames.append(pd.DataFrame({c: [None] for c in formatted.columns}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# -------------------
# UI
# -------------------
st.title("📈 Crypto Option Chain — BTC & ETH (Greeks + OI)")

# Sidebar controls
st.sidebar.header("Controls")
selected_underlying = st.sidebar.selectbox("Underlying", UNDERLYINGS, index=0)
selected_expiry = st.sidebar.selectbox("Expiry", EXPIRIES, index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
show_charts = st.sidebar.checkbox("Show simple charts", value=True)
download_raw = st.sidebar.checkbox("Show download buttons", value=True)

# -------------------
# LIVE PRICES SHOWN HERE
# -------------------
btc_price = get_btc_price()
eth_price = get_eth_price()

if btc_price:
    st.sidebar.metric("BTC Price (USD)", f"${btc_price:,.2f}")
else:
    st.sidebar.metric("BTC Price (USD)", "Error")

if eth_price:
    st.sidebar.metric("ETH Price (USD)", f"${eth_price:,.2f}")
else:
    st.sidebar.metric("ETH Price (USD)", "Error")

# Auto-refresh
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh as _st_autorefresh
        _st_autorefresh(interval=REFRESH_SECONDS * 1000, limit=None, key="autorefresh")
    except:
        pass

# -------------------
# DATA FETCH
# -------------------
with st.spinner("Fetching option chain..."):
    df_chain = fetch_chain_for_underlying(selected_underlying, [selected_expiry])

if df_chain.empty:
    st.warning("No data returned.")
else:
    # ---------------------------------------------------
    # HIGHLIGHT ROWS BETWEEN WHICH CURRENT PRICE FALLS
    # ---------------------------------------------------
    current_price = btc_price if selected_underlying == "BTC" else eth_price
    df_display = df_chain.copy()

    df_display["strike_price"] = pd.to_numeric(df_display["strike_price"], errors="coerce")

    # find nearest strikes
    strikes = df_display["strike_price"].dropna().unique()
    strikes_sorted = sorted(strikes)

    lower = max([s for s in strikes_sorted if s <= current_price], default=None)
    upper = min([s for s in strikes_sorted if s >= current_price], default=None)

    def highlight_row(row):
        if row["strike_price"] == lower or row["strike_price"] == upper:
            return ["background-color: yellow"] * len(row)
        return [""] * len(row)

    st.subheader(f"{selected_underlying} — expiry {selected_expiry}")
    st.dataframe(df_display.style.apply(highlight_row, axis=1).format(na_rep=""))

    # Download CSV
    if download_raw:
        st.download_button(
            label="Download CSV",
            data=df_to_csv_bytes(df_chain),
            file_name=f"option_chain_{selected_underlying}_{selected_expiry}.csv",
            mime="text/csv"
        )

    # Charts
    if show_charts:
        tmp = df_display.dropna(subset=["strike_price"])
        tmp = tmp.sort_values("strike_price")

        chart_df = tmp[["strike_price", "call_oi", "put_oi"]].copy()
        chart_df["call_oi"] = pd.to_numeric(chart_df["call_oi"], errors="coerce").fillna(0)
        chart_df["put_oi"] = pd.to_numeric(chart_df["put_oi"], errors="coerce").fillna(0)

        st.subheader("Open Interest by Strike")
        st.line_chart(chart_df.set_index("strike_price")[["call_oi", "put_oi"]])

st.markdown("---")
st.caption("Data via delta.exchange + Coingecko APIs.")
