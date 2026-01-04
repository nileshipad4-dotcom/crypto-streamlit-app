import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import time
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("⏱ ETH Timestamp Selector")

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "refs/heads/main/data/ETH.csv"
)

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

def extract_timestamps(df, col_idx=14):
    return sorted(
        df.iloc[:, col_idx]
        .astype(str)
        .str[:5]
        .dropna()
        .unique()
        .tolist()
    )

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

# -------------------------------------------------
# AUTO REFRESH EVERY 10 SECONDS (SAFE)
# -------------------------------------------------
st_autorefresh(interval=10_000, key="auto_refresh")

# -------------------------------------------------
# MANUAL REFRESH BUTTON
# -------------------------------------------------
st.button("⏱ Time Refresh")

# -------------------------------------------------
# FETCH DATA
# -------------------------------------------------
try:
    df = fetch_csv_no_cache(CSV_URL)
    st.session_state.timestamps = extract_timestamps(df)
    st.session_state.last_refresh = datetime.utcnow()
except Exception as e:
    st.error(f"❌ Data fetch failed: {e}")

# -------------------------------------------------
# TIMESTAMP DROPDOWNS (ONLY UI)
# -------------------------------------------------
timestamps = st.session_state.timestamps

if timestamps:
    col1, col2 = st.columns(2)

    with col1:
        t1 = st.selectbox("Timestamp 1", timestamps, key="t1")

    with col2:
        t2 = st.selectbox("Timestamp 2", timestamps, key="t2")

# -------------------------------------------------
# LAST REFRESH TIME (ONLY INFO SHOWN)
# -------------------------------------------------
if st.session_state.last_refresh:
    st.caption(
        f"Last refreshed (UTC): "
        f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
    )
