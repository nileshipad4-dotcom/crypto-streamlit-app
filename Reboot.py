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
st.title("🧪 CSV Real-Time Debugger (Hard Reload, No Cache)")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def hard_reset_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def fetch_csv_strong_no_cache(url: str) -> pd.DataFrame:
    """
    Fetch CSV with:
    - hard cache busting
    - new HTTP session
    - no reuse of connections
    """
    cache_buster = int(time.time() * 1000)
    url = f"{url}?cb={cache_buster}"

    with requests.Session() as session:
        r = session.get(
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
    ts_set = set(timestamps)
    probe = now_dt.replace(second=0, microsecond=0)

    for i in range(180):  # look back 3 hours
        s = probe.strftime("%H:%M")
        if s in ts_set:
            return s, i
        probe -= timedelta(minutes=1)

    return None, None

# -------------------------------------------------
# INPUT UI
# -------------------------------------------------
st.subheader("🔗 Paste CSV RAW URL")

csv_url = st.text_input(
    "CSV Raw URL",
    placeholder="https://raw.githubusercontent.com/.../BTC.csv",
    key="csv_url",
)

fetch_btn = st.button("🔄 Fetch CSV NOW (Hard Reload)")

# -------------------------------------------------
# HARD RESET + RERUN
# -------------------------------------------------
if fetch_btn:
    hard_reset_state()
    st.experimental_rerun()

# -------------------------------------------------
# FETCH + DISPLAY
# -------------------------------------------------
if csv_url:
    try:
        df = fetch_csv_strong_no_cache(csv_url)

        st.success("✅ CSV fetched successfully (fresh load, no cache)")

        # --- Metadata ---
        now_ist = get_ist_datetime()
        st.write("🕒 Current IST:", now_ist.strftime("%Y-%m-%d %H:%M:%S"))
        st.write("📦 Rows in CSV:", len(df))
        st.write("📦 Columns in CSV:", list(df.columns))

        # --- Timestamp logic ---
        timestamps = extract_timestamps(df)

        st.write("⏱ Last 10 unique timestamps:", sorted(timestamps)[-10:])

        t1, back_minutes = find_matching_timestamp(timestamps, now_ist)

        if t1:
            st.success(
                f"🎯 Matching timestamp found: {t1} "
                f"(matched {back_minutes} minute(s) back)"
            )
        else:
            st.error("❌ No timestamp matches current IST (even after backtracking)")

        # --- Preview ---
        st.subheader("📄 CSV Preview — LAST 50 ROWS")
        st.dataframe(df.tail(50), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to fetch or parse CSV: {e}")
else:
    st.info("Paste a CSV RAW URL and click **Fetch CSV NOW**")
