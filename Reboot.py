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
st.title("📊 CSV Timestamp Detector (Real-Time, Cache-Free)")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def rotated_time_sort(times, pivot="17:30"):
    pivot_minutes = int(pivot[:2]) * 60 + int(pivot[3:])
    def key(t):
        h, m = map(int, t.split(":"))
        return ((h * 60 + m) - pivot_minutes) % (24 * 60)
    return sorted(times, key=key, reverse=True)

def find_t1_t2(csv_times, now_dt):
    ts_set = set(csv_times)
    sorted_ts = rotated_time_sort(csv_times)

    probe = now_dt.replace(second=0, microsecond=0)
    for _ in range(360):  # search back 6 hours
        s = probe.strftime("%H:%M")
        if s in ts_set:
            idx = sorted_ts.index(s)
            t2 = sorted_ts[idx + 1] if idx + 1 < len(sorted_ts) else None
            return s, t2
        probe -= timedelta(minutes=1)

    return None, None

def fetch_csv_no_cache(url):
    """
    Fetch CSV bypassing ALL HTTP caches
    """
    cache_buster = int(time.time() * 1000)
    url = f"{url}?_cb={cache_buster}"

    r = requests.get(
        url,
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        timeout=20
    )
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
for k in ["df", "timestamps", "t1", "t2", "last_fetch"]:
    if k not in st.session_state:
        st.session_state[k] = None

# -------------------------------------------------
# INPUT
# -------------------------------------------------
st.subheader("🔗 CSV Raw File URL")

csv_url = st.text_input(
    "Paste RAW CSV URL",
    placeholder="https://raw.githubusercontent.com/.../BTC.csv"
)

update = st.button("🔄 Update (Force Fresh Read)")

# -------------------------------------------------
# UPDATE LOGIC (CORE)
# -------------------------------------------------
if update:
    try:
        df = fetch_csv_no_cache(csv_url)

        TIMESTAMP_COL_IDX = 14
        df["timestamp"] = df.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]

        times = (
            df["timestamp"]
            .dropna()
            .unique()
            .tolist()
        )

        if not times:
            st.error("❌ No timestamps found in CSV")
        else:
            now = get_ist_datetime()
            t1, t2 = find_t1_t2(times, now)

            if not t1:
                st.error("❌ No timestamp matches current time window")
            else:
                st.session_state.df = df
                st.session_state.timestamps = rotated_time_sort(times)
                st.session_state.t1 = t1
                st.session_state.t2 = t2
                st.session_state.last_fetch = now.strftime("%H:%M:%S")

                st.success(
                    f"✅ Updated from CSV | T1: {t1} | T2: {t2}"
                )

    except Exception as e:
        st.error(f"❌ Failed to fetch CSV: {e}")

# -------------------------------------------------
# DISPLAY TIMESTAMPS
# -------------------------------------------------
if st.session_state.timestamps:

    st.subheader("⏱ Active Timestamps")

    t1 = st.selectbox(
        "Timestamp 1",
        st.session_state.timestamps,
        index=st.session_state.timestamps.index(st.session_state.t1),
        key="t1"
    )

    t2 = st.selectbox(
        "Timestamp 2",
        st.session_state.timestamps,
        index=st.session_state.timestamps.index(st.session_state.t2)
        if st.session_state.t2 in st.session_state.timestamps else 1,
        key="t2"
    )

    st.caption(
        f"🕒 Current IST: {get_ist_datetime().strftime('%H:%M:%S')} | "
        f"CSV fetched at: {st.session_state.last_fetch}"
    )

# -------------------------------------------------
# DEBUG (IMPORTANT)
# -------------------------------------------------
with st.expander("🔍 Debug: CSV Preview (Top 20 Rows)"):
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(20))
    else:
        st.write("No CSV loaded yet.")
