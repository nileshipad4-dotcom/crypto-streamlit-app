import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import calendar

# =================================================
# PAGE CONFIG — MUST BE FIRST
# =================================================
st.set_page_config(
    page_title="Crypto Option Chain — Max Pain",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================
# SAFE HARD REBOOT MECHANISM
# =================================================
if "do_reboot" not in st.session_state:
    st.session_state.do_reboot = False

# 🔥 perform reboot ONLY at top of next run
if st.session_state.do_reboot:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()

# =================================================
# SIDEBAR CONTROL
# =================================================
st.sidebar.markdown("### ⚙️ App Control")

if st.sidebar.button("🔄 HARD REBOOT APP"):
    st.session_state.do_reboot = True
    st.rerun()

# =================================================
# CONFIG
# =================================================
UNDERLYINGS = ["BTC", "ETH"]
REFRESH_SECONDS = 360
API_BASE = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}

# =================================================
# IST TIME
# =================================================
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_ist_time():
    return get_ist_datetime().strftime("%H:%M:%S")

# =================================================
# EXPIRY LOGIC
# =================================================
def get_expiries():
    ist_now = get_ist_datetime()
    today = ist_now.date()

    latest_valid = (
        today if ist_now.time() <= datetime.strptime("17:30", "%H:%M").time()
        else today + timedelta(days=1)
    )

    expiries = {latest_valid}

    d = today
    while d.weekday() != calendar.FRIDAY:
        d += timedelta(days=1)
    if d >= latest_valid:
        expiries.add(d)

    for week in calendar.monthcalendar(today.year, today.month):
        if week[calendar.FRIDAY]:
            fd = date(today.year, today.month, week[calendar.FRIDAY])
            if fd >= latest_valid:
                expiries.add(fd)

    return sorted(
        [d.strftime("%d-%m-%Y") for d in expiries],
        key=lambda x: datetime.strptime(x, "%d-%m-%Y")
    )

# =================================================
# LIVE PRICE
# =================================================
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    try:
        r = requests.get(API_BASE, timeout=10).json()["result"]
        return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])
    except:
        return None

# =================================================
# FETCH OPTIONS
# =================================================
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

# =================================================
# FORMAT CHAIN
# =================================================
def format_option_chain(df_raw):
    df = df_raw[["strike_price", "contract_type", "mark_price", "oi_contracts"]].copy()
    df[["strike_price", "mark_price", "oi_contracts"]] = df[
        ["strike_price", "mark_price", "oi_contracts"]
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    calls = df[df.contract_type == "call_options"].rename(
        columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}
    )
    puts = df[df.contract_type == "put_options"].rename(
        columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}
    )

    return pd.merge(
        calls[["strike_price", "call_mark", "call_oi"]],
        puts[["strike_price", "put_mark", "put_oi"]],
        on="strike_price",
        how="outer"
    ).fillna(0)

# =================================================
# MAX PAIN
# =================================================
def compute_max_pain(df):
    mp = []
    for i in range(len(df)):
        mp.append(
            (
                -sum(df.call_mark[i:] * df.call_oi[i:])
                + df.strike_price[i] * sum(df.call_oi[:i])
                - sum(df.strike_price[:i] * df.call_oi[:i])
                - sum(df.put_mark[:i] * df.put_oi[:i])
                + sum(df.strike_price[i:] * df.put_oi[i:])
                - df.strike_price[i] * sum(df.put_oi[i:])
            ) / 10000
        )
    df["max_pain"] = mp
    return df

# =================================================
# UI
# =================================================
st.title("📈 Crypto Option Chain — Max Pain")

u = st.sidebar.selectbox("Underlying", UNDERLYINGS)
e = st.sidebar.selectbox("Expiry", get_expiries())

spot = get_delta_price(u)
st.sidebar.metric(f"{u} Spot", f"{spot:,.0f}" if spot else "Error")
st.sidebar.caption(f"IST {get_ist_time()}")

# =================================================
# PROCESS
# =================================================
df = compute_max_pain(format_option_chain(fetch_tickers(u, e)))
df = df.sort_values("strike_price").reset_index(drop=True)
df["Δ max_pain"] = df.max_pain.diff()

df_final = df[["strike_price", "max_pain", "Δ max_pain"]].round(0).astype("Int64")

# =================================================
# DISPLAY
# =================================================
st.subheader(f"{u} — Expiry {e}")
st.caption("🔄 HARD REBOOT clears cache + state safely")

st.dataframe(df_final, use_container_width=True)
