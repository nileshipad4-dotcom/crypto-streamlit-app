import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Live Max Pain Viewer",
    layout="wide"
)

st.title("📊 Live Max Pain (Auto-Loaded from app.py)")

# ------------------------------------------------
# READ SHARED LIVE FILE
# ------------------------------------------------
file_path = "shared/live_max_pain.csv"

if not os.path.exists(file_path):
    st.warning("Waiting for app.py to generate live max pain data…")
    st.stop()

df = pd.read_csv(file_path)

# ------------------------------------------------
# VALIDATE COLUMNS
# ------------------------------------------------
required_cols = ["strike_price", "max_pain"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns in live file: {missing}")
    st.stop()

# ------------------------------------------------
# CLEAN & SORT
# ------------------------------------------------
df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
df["max_pain"] = pd.to_numeric(df["max_pain"], errors="coerce")

df = df.dropna(subset=["strike_price"])
df = df.sort_values("strike_price").reset_index(drop=True)

# ------------------------------------------------
# DISPLAY TABLE
# ------------------------------------------------
st.subheader("Strike vs Max Pain (U Column)")

st.dataframe(
    df[["strike_price", "max_pain"]],
    use_container_width=True
)

st.caption("Data auto-updated by app.py every 30 seconds")

