import streamlit as st
import pandas as pd
import requests
from io import StringIO
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("⏱ Crypto Time Selector")

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "refs/heads/main/data/"
)

PIVOT_TIME = "17:30"
ASSETS = ["BTC", "ETH"]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def fetch_csv_no_cache(url: str) -> pd.DataFrame:
    cache_buster = int(time.time() * 1000)
    url = f"{url}?cb={cache_buster}"

    r = requests.get(
        url,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
        timeout=30,
    )
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def extract_timestamps_desc(df, col_idx=14, pivot=PIVOT_TIME):
    times = (
        df.iloc[:, col_idx]
        .astype(str)
        .str[:5]
        .dropna()
        .unique()
        .tolist()
    )

    def to_minutes(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    pivot_min = to_minutes(pivot)

    def circular_key(t):
        return (pivot_min - to_minutes(t)) % (24 * 60)

    return sorted(times, key=circular_key)


def get_top_timestamp(asset):
    url = f"{BASE_URL}{asset}.csv"
    df = fetch_csv_no_cache(url)
    ts = extract_timestamps_desc(df)
    return ts[0], ts


def to_minutes(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "asset" not in st.session_state:
    st.session_state.asset = None

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# -------------------------------------------------
# INITIAL LOAD LOGIC (RUNS ONCE)
# -------------------------------------------------
if not st.session_state.initialized:
    try:
        btc_top, btc_ts = get_top_timestamp("BTC")
        eth_top, eth_ts = get_top_timestamp("ETH")

        # Compare circular order
        if to_minutes(btc_top) > to_minutes(eth_top):
            st.session_state.asset = "BTC"
            st.session_state.timestamps = btc_ts
        else:
            st.session_state.asset = "ETH"
            st.session_state.timestamps = eth_ts

        st.session_state.initialized = True

    except Exception as e:
        st.error(f"❌ Initial load failed: {e}")

# -------------------------------------------------
# TOP CONTROL BAR
# -------------------------------------------------
bar_col1, bar_col2 = st.columns([1, 8])

with bar_col1:
    st.selectbox(
        "",
        ASSETS,
        key="asset",
        label_visibility="collapsed",
    )

with bar_col2:
    refresh = st.button("⏱")

# -------------------------------------------------
# MANUAL REFRESH
# -------------------------------------------------
if refresh:
    try:
        url = f"{BASE_URL}{st.session_state.asset}.csv"
        df = fetch_csv_no_cache(url)
        st.session_state.timestamps = extract_timestamps_desc(df)
    except Exception as e:
        st.error(f"❌ Data fetch failed: {e}")

# -------------------------------------------------
# TIMESTAMP DROPDOWNS
# -------------------------------------------------
timestamps = st.session_state.timestamps

if timestamps:
    default_1 = 0
    default_2 = 1 if len(timestamps) > 1 else 0

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "",
            timestamps,
            index=default_1,
            key="t1",
            label_visibility="collapsed",
        )

    with col2:
        st.selectbox(
            "",
            timestamps,
            index=default_2,
            key="t2",
            label_visibility="collapsed",
        )
