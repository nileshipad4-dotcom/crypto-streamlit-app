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
EXPIRIES = ["12-12-2025"]
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
# LIVE PRICE FROM DELTA EXCHANGE
# -------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    """
    Fetch asset index price from Delta Exchange tickers API.
    symbol = BTC or ETH
    """
    try:
        url = "https://api.india.delta.exchange/v2/tickers"
        r = requests.get(url, timeout=10).json()
        results = r.get("result", [])

        if symbol == "BTC":
            ticker = next((x for x in results if x.get("symbol") == "BTCUSD"), None)
        elif symbol == "ETH":
            ticker = next((x for x in results if x.get("symbol") == "ETHUSD"), None)
        else:
            return None

        if ticker and "mark_price" in ticker:
            return float(ticker["mark_price"])
        return None
    except:
        return None


# -------------------
# HELPERS
# -------------------
@st.cache_data(ttl=60)
def fetch_tickers(underlying: str, expiry: str) -> pd.DataFrame:
    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry}"
    )

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
# MAX PAIN CALCULATION
# -------------------
def compute_max_pain(df):
    df = df.copy()

    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    n = len(df)

    Q, R, S, T, U = [], [], [], [], []

    for i in range(n):
        q = -sum(A[i:] * B[i:])
        r = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        s = -sum(M[:i] * L[:i])
        t = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        u = round((q + r + s + t) / 10000)

        Q.append(q)
        R.append(r)
        S.append(s)
        T.append(t)
        U.append(u)

    df["Q_call_cost"] = Q
    df["R_call_intrinsic"] = R
    df["S_put_cost"] = S
    df["T_put_intrinsic"] = T
    df["max_pain"] = U

    return df


# -------------------
# UI
# -------------------
st.title("📈 Crypto Option Chain — BTC & ETH (Greeks + OI + Max Pain)")

st.sidebar.header("Controls")
selected_underlying = st.sidebar.selectbox("Underlying", UNDERLYINGS)
selected_expiry = st.sidebar.selectbox("Expiry", EXPIRIES)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
show_charts = st.sidebar.checkbox("Show charts", value=True)
download_raw = st.sidebar.checkbox("Download CSV", value=True)

# LIVE PRICE FROM DELTA
price_btc = get_delta_price("BTC")
price_eth = get_delta_price("ETH")

st.sidebar.metric("BTC Price (Delta)", f"${price_btc:,.2f}" if price_btc else "Error")
st.sidebar.metric("ETH Price (Delta)", f"${price_eth:,.2f}" if price_eth else "Error")

if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
    except:
        pass


# -------------------
# FETCH DATA
# -------------------
with st.spinner("Fetching option chain..."):
    df_chain = fetch_chain_for_underlying(selected_underlying, [selected_expiry])

if df_chain.empty:
    st.warning("No data returned.")
else:
    price = price_btc if selected_underlying == "BTC" else price_eth
    df_display = df_chain.copy()

    df_display["strike_price"] = pd.to_numeric(df_display["strike_price"], errors="coerce")

    # 🔥 Add max pain columns
    df_display = compute_max_pain(df_display)

    # Highlight ATM
    strikes = df_display["strike_price"].dropna().unique()
    strikes_sorted = sorted(strikes)

    lower = max([s for s in strikes_sorted if price and s <= price], default=None)
    upper = min([s for s in strikes_sorted if price and s >= price], default=None)

    def highlight_row(row):
        if row["strike_price"] in (lower, upper):
            return ["background-color: yellow"] * len(row)
        return [""] * len(row)

    st.subheader(f"{selected_underlying} — expiry {selected_expiry}")
    st.dataframe(df_display.style.apply(highlight_row, axis=1))

    # Download CSV
    if download_raw:
        st.download_button(
            "Download CSV",
            df_to_csv_bytes(df_chain),
            file_name=f"option_chain_{selected_underlying}.csv",
            mime="text/csv"
        )

    # Charts
    if show_charts:
        tmp = df_display.dropna(subset=["strike_price"]).sort_values("strike_price")

        chart_df = tmp[["strike_price", "call_oi", "put_oi", "max_pain"]].copy()
        chart_df["call_oi"] = pd.to_numeric(chart_df["call_oi"], errors="coerce").fillna(0)
        chart_df["put_oi"] = pd.to_numeric(chart_df["put_oi"], errors="coerce").fillna(0)

        st.subheader("Open Interest by Strike")
        st.line_chart(chart_df.set_index("strike_price")[["call_oi", "put_oi"]])

        st.subheader("Max Pain by Strike")
        st.line_chart(chart_df.set_index("strike_price")[["max_pain"]])

st.markdown("---")
st.caption("Data via Delta Exchange APIs.")
