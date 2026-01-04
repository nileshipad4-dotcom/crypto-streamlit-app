# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import calendar

# =================================================
# 🔴 HARD REBOOT BUTTON (TOP-LEVEL)
# =================================================
if "reboot_requested" not in st.session_state:
    st.session_state.reboot_requested = False

def reboot_app():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.experimental_rerun()

with st.sidebar:
    st.markdown("### ⚙️ App Control")
    if st.button("🔄 HARD REBOOT APP"):
        reboot_app()

# =================================================
# CONFIG
# =================================================
UNDERLYINGS = ["BTC", "ETH"]
REFRESH_SECONDS = 360
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
# EXPIRY LOGIC
# -------------------
def get_expiries():
    ist_now = get_ist_datetime()
    today = ist_now.date()

    if ist_now.time() <= datetime.strptime("17:30", "%H:%M").time():
        latest_valid = today
    else:
        latest_valid = today + timedelta(days=1)

    expiries = {latest_valid}

    d = today
    while d.weekday() != calendar.FRIDAY:
        d += timedelta(days=1)
    if d >= latest_valid:
        expiries.add(d)

    year, month = today.year, today.month
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[calendar.FRIDAY]:
            fd = date(year, month, week[calendar.FRIDAY])
            if fd >= latest_valid:
                expiries.add(fd)

    for i in [1, 2]:
        m = month + i
        y = year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        cal = calendar.monthcalendar(y, m)
        last_friday = max(
            w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY]
        )
        expiries.add(date(y, m, last_friday))

    return sorted(
        [d.strftime("%d-%m-%Y") for d in expiries],
        key=lambda x: datetime.strptime(x, "%d-%m-%Y")
    )

# -------------------
# LIVE PRICE
# -------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        sym = f"{symbol}USD"
        return float(next(x for x in r if x["symbol"] == sym)["mark_price"])
    except:
        return None

# -------------------
# FETCH OPTION DATA
# -------------------
@st.cache_data(ttl=60)
def fetch_tickers(underlying, expiry):
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
def format_option_chain(df_raw):
    cols = ["strike_price", "contract_type", "mark_price", "oi_contracts"]
    for c in cols:
        if c not in df_raw.columns:
            df_raw[c] = 0

    df = df_raw[cols].copy()
    for c in ["strike_price", "mark_price", "oi_contracts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    calls = df[df["contract_type"] == "call_options"].rename(
        columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}
    ).drop(columns="contract_type")

    puts = df[df["contract_type"] == "put_options"].rename(
        columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}
    ).drop(columns="contract_type")

    return pd.merge(calls, puts, on="strike_price", how="outer").fillna(0)

# -------------------
# MAX PAIN
# -------------------
def compute_max_pain(df):
    A, B = df["call_mark"].values, df["call_oi"].values
    G = df["strike_price"].values
    L, M = df["put_oi"].values, df["put_mark"].values

    U = []
    for i in range(len(df)):
        U.append(
            (
                -sum(A[i:] * B[i:])
                + G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
                - sum(M[:i] * L[:i])
                + sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
            ) / 10000
        )
    df["max_pain"] = U
    return df

# =================================================
# UI
# =================================================
st.title("📈 Crypto Option Chain — Max Pain")

selected_underlying = st.sidebar.selectbox("Underlying", UNDERLYINGS)
selected_expiry = st.sidebar.selectbox("Expiry", get_expiries())

auto_refresh = st.sidebar.checkbox("Auto-refresh", True)

spot = get_delta_price(selected_underlying)
st.sidebar.metric(
    f"{selected_underlying} Spot",
    f"{spot:,.0f}" if spot else "Error"
)

st.sidebar.caption(f"IST Time: {get_ist_time()}")

if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
    except:
        pass

# =================================================
# PROCESS
# =================================================
raw = fetch_tickers(selected_underlying, selected_expiry)
df = compute_max_pain(format_option_chain(raw))
df = df.sort_values("strike_price").reset_index(drop=True)
df["Δ max_pain"] = df["max_pain"].diff()

df_final = df[["strike_price", "max_pain", "Δ max_pain"]].round(0).astype("Int64")

# -------------------
# HIGHLIGHTS
# -------------------
mp_strike = df_final.loc[df_final["max_pain"].idxmin(), "strike_price"]
low = df_final[df_final["strike_price"] <= spot]["strike_price"].max()
high = df_final[df_final["strike_price"] >= spot]["strike_price"].min()

def highlight(row):
    if row["strike_price"] == mp_strike:
        return ["background-color:#960018"] * len(row)
    if row["strike_price"] in (low, high):
        return ["background-color:#4B0082"] * len(row)
    return [""] * len(row)

# -------------------
# DISPLAY
# -------------------
st.subheader(f"{selected_underlying} — Expiry {selected_expiry}")
st.caption(f"IST {get_ist_time()} • Spot band (indigo) • Max Pain (red)")
st.dataframe(df_final.style.apply(highlight, axis=1), use_container_width=True)

st.caption("Intraday-aware expiry logic • Delta Exchange • HARD REBOOT clears cache + state")
