# app.py
import streamlit as st
import pandas as pd
import requests
import os

# -------------------
# CONFIG
# -------------------
EXPIRY = "15-12-2025"
UNDERLYINGS = ["BTC", "ETH"]
REFRESH_SECONDS = 30

API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}

st.set_page_config(
    page_title="Live Max Pain",
    layout="wide"
)

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
def fetch_option_chain(underlying, expiry):
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
# FORMAT OPTION CHAIN (SAFE)
# -------------------
def format_chain(df):

    df = df[
        ["strike_price", "contract_type", "mark_price", "oi_contracts"]
    ].copy()

    for col in ["strike_price", "mark_price", "oi_contracts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    calls = df[df["contract_type"] == "call_options"].copy()
    puts  = df[df["contract_type"] == "put_options"].copy()

    calls = calls.rename(columns={
        "mark_price": "call_mark",
        "oi_contracts": "call_oi"
    }).drop(columns="contract_type")

    puts = puts.rename(columns={
        "mark_price": "put_mark",
        "oi_contracts": "put_oi"
    }).drop(columns="contract_type")

    merged = pd.merge(calls, puts, on="strike_price", how="inner")
    return merged.sort_values("strike_price").reset_index(drop=True)


# -------------------
# MAX PAIN (U COLUMN)
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
        U.append(round((Q + R + S + T) / 10000))

    df["max_pain"] = U
    return df


# -------------------
# UI
# -------------------
st.title("📊 Live Max Pain (Strike vs U Column)")

underlying = st.selectbox("Select Underlying", UNDERLYINGS)

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
except:
    pass

# Live price
price = get_delta_price(underlying)
st.metric(
    f"{underlying} Price (Delta)",
    f"{price:,.2f}" if price else "Error"
)

with st.spinner("Fetching option chain..."):
    raw = fetch_option_chain(underlying, EXPIRY)

if raw.empty:
    st.warning("No data returned from Delta Exchange.")
    st.stop()

df_chain = format_chain(raw)

if df_chain.empty:
    st.error("Option chain formatting failed.")
    st.stop()

df_chain = compute_max_pain(df_chain)

# -------------------
# FINAL TABLE (ONLY REQUIRED COLUMNS)
# -------------------
df_mp = df_chain[["strike_price", "max_pain"]].copy()

# Identify lower & upper strike around price
lower_strike = None
upper_strike = None

if price is not None:
    strikes = df_mp["strike_price"].values
    lower = [s for s in strikes if s <= price]
    upper = [s for s in strikes if s >= price]

    if lower:
        lower_strike = max(lower)
    if upper:
        upper_strike = min(upper)

# -------------------
# ROW HIGHLIGHTING
# -------------------
def highlight_row(row):
    if row["strike_price"] == lower_strike:
        return ["background-color: lightgreen"] * len(row)
    if row["strike_price"] == upper_strike:
        return ["background-color: lightcoral"] * len(row)
    return [""] * len(row)

# -------------------
# SAVE FOR OTHER APPS
# -------------------
os.makedirs("shared", exist_ok=True)
df_mp.to_csv("shared/live_max_pain.csv", index=False)

# -------------------
# DISPLAY
# -------------------
st.subheader(f"Max Pain — {underlying} | Expiry {EXPIRY}")

st.dataframe(
    df_mp.style.apply(highlight_row, axis=1),
    use_container_width=True
)

# Download
st.download_button(
    "Download Max Pain CSV",
    data=df_mp.to_csv(index=False),
    file_name=f"max_pain_{underlying}_{EXPIRY}.csv",
    mime="text/csv"
)

st.caption("🟢 Lower strike | 🔴 Upper strike • Auto-updated every 30s")
