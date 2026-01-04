import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("⏱ Crypto Timestamp Selector")

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "refs/heads/main/data/"
)

PIVOT_TIME = "17:30"  # circular sort anchor

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
    """
    Circular descending order starting from pivot (17:30),
    wrapping across midnight correctly.
    """
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

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "asset" not in st.session_state:
    st.session_state.asset = "ETH"

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

# -------------------------------------------------
# ASSET SELECTOR
# -------------------------------------------------
st.selectbox(
    "Select Asset",
    ["BTC", "ETH"],
    key="asset",
)

csv_url = f"{BASE_URL}{st.session_state.asset}.csv"

# -------------------------------------------------
# MANUAL REFRESH BUTTON
# -------------------------------------------------
refresh = st.button("⏱ Time Refresh")

# -------------------------------------------------
# FETCH DATA (ONLY ON BUTTON CLICK)
# -------------------------------------------------
if refresh:
    try:
        df = fetch_csv_no_cache(csv_url)
        st.session_state.timestamps = extract_timestamps_desc(df)
        st.session_state.last_refresh = datetime.utcnow()
    except Exception as e:
        st.error(f"❌ Data fetch failed: {e}")

# -------------------------------------------------
# TIMESTAMP DROPDOWNS
# -------------------------------------------------
timestamps = st.session_state.timestamps

if timestamps:
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Timestamp 1",
            timestamps,
            key="t1",
        )

    with col2:
        st.selectbox(
            "Timestamp 2",
            timestamps,
            key="t2",
        )

# -------------------------------------------------
# LAST REFRESH TIME (ONLY INFO SHOWN)
# -------------------------------------------------
if st.session_state.last_refresh:
    st.caption(
        "Last refreshed (UTC): "
        f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
    )
