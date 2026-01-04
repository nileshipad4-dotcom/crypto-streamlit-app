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
st.title("🧪 CSV Real-Time Debugger (Correct Streamlit Model)")

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

def extract_timestamps(df, col_idx=14):
    return (
        df.iloc[:, col_idx]
        .astype(str)
        .str[:5]
        .dropna()
        .unique()
        .tolist()
    )

def find_matching_timestamp(timestamps, now_dt):
    probe = now_dt.replace(second=0, microsecond=0)
    ts_set = set(timestamps)

    for i in range(180):
        s = probe.strftime("%H:%M")
        if s in ts_set:
            return s, i
        probe -= timedelta(minutes=1)

    return None, None

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None

# -------------------------------------------------
# INPUT UI
# -------------------------------------------------
st.subheader("🔗 Paste CSV RAW URL")

csv_url = st.text_input(
    "CSV Raw URL",
    placeholder="https://raw.githubusercontent.com/.../BTC.csv",
)

fetch_btn = st.button("🔄 Fetch CSV NOW (Fresh Load)")

# -------------------------------------------------
# FETCH (ONLY WHEN BUTTON IS PRESSED)
# -------------------------------------------------
if fetch_btn:
    try:
        df = fetch_csv_strong_no_cache(csv_url)
        st.session_state.df = df
        st.session_state.last_fetch = datetime.utcnow()
        st.success("✅ CSV fetched fresh from source")

    except Exception as e:
        st.error(f"❌ Fetch failed: {e}")

# -------------------------------------------------
# DISPLAY (USES SESSION STATE)
# -------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    now_ist = get_ist_datetime()
    st.write("🕒 Current IST:", now_ist.strftime("%Y-%m-%d %H:%M:%S"))
    st.write("🕒 Last Fetch UTC:", st.session_state.last_fetch)
    st.write("📦 Rows:", len(df))
    st.write("📦 Columns:", list(df.columns))

    timestamps = extract_timestamps(df)
    st.write("⏱ Last 10 timestamps:", sorted(timestamps)[-10:])

    t1, back_minutes = find_matching_timestamp(timestamps, now_ist)

    if t1:
        st.success(f"🎯 Matching timestamp: {t1} ({back_minutes} min back)")
    else:
        st.error("❌ No timestamp match found")

    st.subheader("📄 CSV Preview — LAST 50 ROWS")
    st.dataframe(df.tail(50), use_container_width=True)

else:
    st.info("Click **Fetch CSV NOW** to load data")
