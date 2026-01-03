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
# ✅ CORRECT EXPIRY FETCH (DELTA-SAFE)
# -------------------
@st.cache_data(ttl=300)
def get_valid_expiries(underlying: str):
    """
    Delta crypto options:
    - expire intraday
    - today's expiry is NOT tradable
    Rule: show only dates strictly AFTER today
    """
    r = requests.get(API_BASE, timeout=20).json().get("result", [])

    today = datetime.utcnow().date()
    expiries = set()

    for x in r:
        if x.get("underlying_asset", {}).get("symbol") != underlying:
            continue

        expiry = x.get("expiry_date")
        if not expiry:
            continue

        try:
            expiry_date = datetime.strptime(expiry, "%d-%m-%Y").date()
        except:
            continue

        # ✅ STRICTLY FUTURE ONLY
        if expiry_date > today:
            expiries.add(expiry)

    return sorted(expiries, key=lambda d: datetime.strptime(d, "%d-%m-%Y"))

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

expiry_list = get_valid_expiries(selected_underlying)

if not expiry_list:
    st.error("No valid future expiries available on Delta.")
    st.stop()

selected_expiry = st.sidebar.selectbox(
    "Expiry (Live)",
    expiry_list,
    index=0
)

auto_refresh = st.sidebar.checkbox("Auto-refresh", True)

price_btc = get_delta_price("BTC")
price_eth = get_delta_price("ETH")
spot_price = price_btc if selected_underlying == "BTC" else price_eth

st.sidebar.metric(
    f"{selected_underlying} Spot",
    f"{spot_price:,.0f}" if spot_price else "Error"
)

# -------------------
# PROCESS
# -------------------
raw = fetch_tickers(selected_underlying, selected_expiry)
df = format_option_chain(raw)
df = compute_max_pain(df)

df = df.sort_values("strike_price").reset_index(drop=True)
df["Δ max_pain"] = df["max_pain"].diff()
df["Δ² max_pain"] = df["Δ max_pain"].diff()

df_final = (
    df[["strike_price", "max_pain", "Δ max_pain", "Δ² max_pain"]]
    .round(0)
    .astype("Int64")
)

# -------------------
# HIGHLIGHTS
# -------------------
max_pain_strike = df_final.loc[df_final["max_pain"].idxmin(), "strike_price"]

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
st.caption(f"Only FUTURE expiries shown • Spot range (indigo) • Max Pain (red) • IST {get_ist_time()}")

styled_df = df_final.style.apply(highlight_rows, axis=1)
st.dataframe(styled_df, use_container_width=True)
