import streamlit as st
import pandas as pd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📄 CSV Column Inspector")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
UNDERLYING = st.sidebar.selectbox("Underlying", ["BTC", "ETH"])
CSV_PATH = f"data/{UNDERLYING}.csv"

# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------
try:
    df_raw = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# -------------------------------------------------
# SIDEBAR DEBUG OUTPUT
# -------------------------------------------------
st.sidebar.subheader("CSV Structure")

st.sidebar.write("Total columns:", df_raw.shape[1])

st.sidebar.write("Index → Column Name")
st.sidebar.write(list(enumerate(df_raw.columns)))

# -------------------------------------------------
# PREVIEW DATA (OPTIONAL)
# -------------------------------------------------
st.subheader("CSV Preview (first 5 rows)")
st.dataframe(df_raw.head(), use_container_width=True)
