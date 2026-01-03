# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# -------------------
# CONFIG
# -------------------
UNDERLYINGS = ["BTC", "ETH"]
REFRESH_SECONDS = 60
API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}

st.set_page_config(
    page_title="Crypto Option Chain — Max Pain",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# IST TIME
# -------------------
def get_ist_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M:%S")

# -------------------
# EXPIRIES
# -------------------
def get_expiries(months=3):
    today = datetime.utcnow().date()
    end = today + timedelta(days=months * 31)

    expiries = set()

    # upcoming expiry (next available date)
    expiries.add(today.strftime("%d-%m-%Y"))

    d = today
    while d <= end:
        if d.weekday() == 4:  # Friday
            expiries.add(d.strftime("%d-%m-%Y"))
        d += timedelta(days=1)

    return sorted(expiries, key=lambda x: datetime.strptime(x, "%d-%m-%Y"))

# -------------------
# LIVE PRICE FROM DELTA
# -------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        sym = "BTCUSD" if symbol == "BTC" else "ETHUSD"
        ticker = next(x for x in r if x["symbol"] == sym)
        return float(ticker["mark_price"])
    except:
        return None

# -------------------
# FETCH OPTION DATA
# -------------------
@st.cache_data(ttl=60)
def fetch_tickers(underlying: str, expiry: str) -> pd.DataFrame:
    url = (
        f"{API_BASE}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry}"
    )
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.json_normalize(r.json().get("result", []))

# -------------------
# FORMAT OPTION CHAIN
# -------------------
def format_option_chain(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["strike_price", "contract_type", "mark_price", "oi_contracts"]
    for c in cols:
        if c not in df_raw.columns:
            df_raw[c] = 0

    df = df_raw[cols].copy()
    for c in ["strike_price", "mark_price", "oi_contracts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    calls = df[df["contract_type"] == "call_options"]
    puts = df[df["contract_type"] == "put_options"]

    calls = calls.rename(columns={
        "mark_price": "call_mark",
        "oi_contracts": "call_oi"
    }).drop(columns="contract_type")

    puts = puts.rename(columns={
        "mark_price": "put_mark",
        "oi_contracts": "put_oi"
    }).drop(columns="contract_type")

    return pd.merge(calls, puts, on="strike_price", how="outer").fillna(0)

# -------------------
# MAX PAIN
# -------------------
def compute_max_pain(df):
    A = df["call_mark"].values
    B = df["call_oi"].values
    G = df["strike_price"].values
    L = df["put_oi"].values
    M = df["put_mark"].values

    U = []
    for i in range(len(df)):
        Q = -sum(A[i:] * B[i:])
        R = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        S = -sum(M[:i] * L[:i])
        T = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        U.append((Q + R + S + T) / 10000)

    df["max_pain"] = U
    return df

# -------------------
# UI
# -------------------
st.title("📈 Crypto Option Chain — Max Pain")

selected_underlying = st.sidebar.selectbox("Underlying", UNDERLYINGS)

expiry_list = get_expiries()
selected_expiry = st.sidebar.selectbox("Expiry", expiry_list)

auto_refresh = st.sidebar.checkbox("Auto-refresh", True)

price_btc = get_delta_price("BTC")
price_eth = get_delta_price("ETH")
spot_price = price_btc if selected_underlying == "BTC" else price_eth

st.sidebar.metric(
    f"{selected_underlying} Spot",
    f"{spot_price:,.0f}" if spot_price else "Error"
)

if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
    except:
        pass

# -------------------
# PROCESS
# -------------------
raw = fetch_tickers(selected_underlying, selected_expiry)
df = format_option_chain(raw)
df = compute_max_pain(df)

df = df.sort_values("strike_price").reset_index(drop=True)
df["Δ max_pain"] = df["max_pain"].diff()
df["Δ² max_pain"] = df["Δ max_pain"].diff()

# Final table (integers)
df_final = (
    df[["strike_price", "max_pain", "Δ max_pain", "Δ² max_pain"]]
    .round(0)
    .astype("Int64")
)

# -------------------
# HIGHLIGHTS
# -------------------
max_pain_strike = df_final.loc[df_final["max_pain"].idxmin(), "strike_price"]

lower_strike = None
upper_strike = None
if spot_price:
    lower_strike = df_final[df_final["strike_price"] <= spot_price]["strike_price"].max()
    upper_strike = df_final[df_final["strike_price"] >= spot_price]["strike_price"].min()

def highlight_rows(row):
    if row["strike_price"] == max_pain_strike:
        return ["background-color: #f8d7da"] * len(row)  # red
    if row["strike_price"] in [lower_strike, upper_strike]:
        return ["background-color: #e0e7ff"] * len(row)  # indigo
    return [""] * len(row)

# -------------------
# DISPLAY
# -------------------
st.subheader(f"{selected_underlying} — Expiry {selected_expiry}")
st.caption(f"Spot range highlighted (indigo) • Max Pain (red) • IST {get_ist_time()}")

styled_df = df_final.style.apply(highlight_rows, axis=1)
st.dataframe(styled_df, use_container_width=True)

st.caption("Max Pain ladder • Δ and Δ² show curvature of pain distribution • Delta Exchange")
