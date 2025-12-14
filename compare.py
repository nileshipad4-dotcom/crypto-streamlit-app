import streamlit as st
import pandas as pd

st.title("📊 Live Max Pain (U Column)")

# --------------------------------------------------------
# UPLOAD LIVE DATA FROM app.py
# --------------------------------------------------------
st.subheader("Upload Live Max Pain Snapshot")

live_file = st.file_uploader(
    "Upload CSV exported from app.py (must contain strike_price & max_pain)",
    type=["csv"]
)

if live_file is None:
    st.info("Please upload the live snapshot CSV from app.py")
    st.stop()

# --------------------------------------------------------
# LOAD & CLEAN DATA
# --------------------------------------------------------
df = pd.read_csv(live_file)

required_cols = ["strike_price", "max_pain"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
df["max_pain"] = pd.to_numeric(df["max_pain"], errors="coerce")

df = df[["strike_price", "max_pain"]].dropna(subset=["strike_price"])

df = df.sort_values("strike_price").reset_index(drop=True)

# --------------------------------------------------------
# DISPLAY TABLE
# --------------------------------------------------------
st.subheader("Live Max Pain by Strike")

st.dataframe(
    df,
    use_container_width=True
)
