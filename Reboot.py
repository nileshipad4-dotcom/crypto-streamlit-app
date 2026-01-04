import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 CSV-based Timestamp Selector (Live Update)")

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

def find_t1_t2_from_csv(timestamps, now_dt):
    """
    timestamps: list of HH:MM strings from CSV
    returns (t1, t2)
    """
    ts_set = set(timestamps)
    sorted_ts = rotated_time_sort(timestamps)

    probe = now_dt.replace(second=0, microsecond=0)

    for _ in range(300):  # look back up to 5 hours
        probe_str = probe.strftime("%H:%M")
        if probe_str in ts_set:
            idx = sorted_ts.index(probe_str)
            t2 = sorted_ts[idx + 1] if idx + 1 < len(sorted_ts) else None
            return probe_str, t2
        probe -= timedelta(minutes=1)

    return None, None

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "t1" not in st.session_state:
    st.session_state.t1 = None

if "t2" not in st.session_state:
    st.session_state.t2 = None

if "df_csv" not in st.session_state:
    st.session_state.df_csv = None

# -------------------------------------------------
# CSV URL INPUT
# -------------------------------------------------
st.subheader("🔗 Enter CSV Raw File URL")

csv_url = st.text_input(
    "CSV Raw URL (GitHub / S3 / public link)",
    placeholder="https://raw.githubusercontent.com/....csv"
)

update_clicked = st.button("🔄 Update")

# -------------------------------------------------
# UPDATE LOGIC (CORE)
# -------------------------------------------------
if update_clicked:
    try:
        # Read CSV fresh every time
        df = pd.read_csv(csv_url)

        # Extract timestamps (assumes HH:MM format)
        TIMESTAMP_COL_IDX = 14
        df["timestamp"] = df.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5]

        timestamps = (
            df["timestamp"]
            .dropna()
            .unique()
            .tolist()
        )

        if not timestamps:
            st.error("No timestamps found in CSV")
        else:
            now_ist = get_ist_datetime()
            t1, t2 = find_t1_t2_from_csv(timestamps, now_ist)

            if not t1:
                st.error("No matching timestamp found relative to current time")
            else:
                st.session_state.df_csv = df
                st.session_state.timestamps = rotated_time_sort(timestamps)
                st.session_state.t1 = t1
                st.session_state.t2 = t2

                st.success(
                    f"Timestamps updated → T1: {t1} | T2: {t2}"
                )

    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# -------------------------------------------------
# TIMESTAMP DROPDOWNS
# -------------------------------------------------
if st.session_state.timestamps:

    st.subheader("⏱ Timestamp Selection")

    t1 = st.selectbox(
        "Timestamp 1 (Auto-detected)",
        st.session_state.timestamps,
        index=st.session_state.timestamps.index(st.session_state.t1)
        if st.session_state.t1 in st.session_state.timestamps else 0,
        key="t1_select"
    )

    t2 = st.selectbox(
        "Timestamp 2 (Previous)",
        st.session_state.timestamps,
        index=st.session_state.timestamps.index(st.session_state.t2)
        if st.session_state.t2 in st.session_state.timestamps else 1,
        key="t2_select"
    )

    st.caption(
        f"🕒 Current IST: {get_ist_datetime().strftime('%H:%M')} | "
        f"CSV rows loaded: {len(st.session_state.df_csv)}"
    )

# -------------------------------------------------
# DEBUG VIEW (OPTIONAL BUT USEFUL)
# -------------------------------------------------
with st.expander("🔍 Debug: CSV Preview"):
    if st.session_state.df_csv is not None:
        st.dataframe(st.session_state.df_csv.head(20))
    else:
        st.write("No CSV loaded yet.")
