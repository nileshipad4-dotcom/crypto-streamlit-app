import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("⏱ Crypto Timestamp Viewer")

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
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def fetch_csv_strong_no_cache(url: str) -> pd.DataFrame:
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

def extract_all_timestamps(df, col_idx=14):
    ts = (
        df.iloc[:, col_idx]
        .astype(str)
        .str[:5]
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(ts)

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None

# -------------------------------------------------
# UI — REFRESH BUTTON
# -------------------------------------------------
st.subheader("📡 Data Source: ETH.csv (GitHub Raw)")

refresh_btn = st.button("⏱ Time Refresh")

# -------------------------------------------------
# FETCH ON BUTTON CLICK
# -------------------------------------------------
if refresh_btn:
    try:
        df = fetch_csv_strong_no_cache(CSV_URL)
        st.session_state.df = df
        st.session_state.timestamps = extract_all_timestamps(df)
        st.session_state.last_fetch = datetime.utcnow()
        st.success("✅ Data refreshed successfully")

    except Exception as e:
        st.error(f"❌ Fetch failed: {e}")

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    timestamps = st.session_state.timestamps

    now_ist = get_ist_datetime()
    st.write("🕒 Current IST:", now_ist.strftime("%Y-%m-%d %H:%M:%S"))
    st.write("🕒 Last Refresh (UTC):", st.session_state.last_fetch)
    st.write("📦 Rows:", len(df))
    st.write("📦 Columns:", list(df.columns))

    st.divider()

    # -------------------------------------------------
    # TIMESTAMP DROPDOWNS (TWO)
    # -------------------------------------------------
    st.subheader("⏰ Select Timestamps")

    col1, col2 = st.columns(2)

    with col1:
        t1 = st.selectbox(
            "Time 1",
            timestamps,
            index=0 if timestamps else None,
        )

    with col2:
        t2 = st.selectbox(
            "Time 2",
            timestamps,
            index=len(timestamps) - 1 if timestamps else None,
        )

    st.info(f"Selected Times → ⏱ {t1}  |  ⏱ {t2}")

    # -------------------------------------------------
    # PREVIEW
    # -------------------------------------------------
    st.subheader("📄 CSV Preview — LAST 50 ROWS")
    st.dataframe(df.tail(50), use_container_width=True)

else:
    st.info("Click **⏱ Time Refresh** to load ETH data")
