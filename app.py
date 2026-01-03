# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import calendar

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
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_ist_time():
    return get_ist_datetime().strftime("%H:%M:%S")

# -------------------
# ✅ FINAL EXPIRY LOGIC (NO EXPIRED, NO LEAKS)
# -------------------
def get_expiries():
    ist_now = get_ist_datetime()
    today = ist_now.date()

    # ---- latest valid expiry ----
    if ist_now.time() <= datetime.strptime("17:30", "%H:%M").time():
        latest_valid = today
    else:
        latest_valid = today + timedelta(days=1)

    expiries = set()
    expiries.add(latest_valid)

    # ---- immediate upcoming Friday ----
    d = today
    while d.weekday() != calendar.FRIDAY:
        d += timedelta(days=1)
    if d >= latest_valid:
        expiries.add(d)

    # ---- all future Fridays of current month ----
    year, month = today.year, today.month
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[calendar.FRIDAY] != 0:
            fd = date(year, month, week[calendar.FRIDAY])
            if fd >= latest_valid:
                expiries.add(fd)

    # ---- last Fridays of next 2 months ----
    for i in [1, 2]:
        m = month + i
        y = year + (m - 1) // 12
        m = ((m - 1) % 12) + 1

        cal = calendar.monthcalendar(y, m)
        last_friday = max(
            week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0
        )
        lf = date(y, m, last_friday)
        if lf >= latest_valid:
            expiries.add(lf)

    return sorted(
        [d.strftime("%d-%m-%Y") for d in expiries],
        key=lambda x: datetime.strptime(x, "%d-%m-%Y")
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

st.sidebar.caption(f"IST Time: {get_ist_time()}")

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
        return ["background-color: #960018"] * len(row)  # red
    if row["strike_price"] in [lower_strike, upper_strike]:
        return ["background-color: #4B0082"] * len(row)  # indigo
    return [""] * len(row)

# -------------------
# DISPLAY
# -------------------
st.subheader(f"{selected_underlying} — Expiry {selected_expiry}")
st.caption(f"IST {get_ist_time()} • Spot range (indigo) • Max Pain (red)")

styled_df = df_final.style.apply(highlight_rows, axis=1)
st.dataframe(styled_df, use_container_width=True)

st.caption("Expiry logic: intraday-aware • no expired dates • immediate Friday ensured • Delta Exchange")

