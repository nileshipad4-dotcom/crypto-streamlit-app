import streamlit as st
import pandas as pd

st.title("📊 Call OI Change Comparison Tool")

# ---------------------
# LOAD DATA
# ---------------------
underlying = st.selectbox("Select Underlying", ["BTC", "ETH"])

file_path = f"data/{underlying}.csv"
df = pd.read_csv(file_path)

# Ensure numeric
df["call_oi"] = pd.to_numeric(df["call_oi"], errors="coerce")
df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")

# ---------------------
# TIME SELECTION
# ---------------------
timestamps = sorted(df["timestamp_IST"].unique())

t1 = st.selectbox("Select Time 1", timestamps)
t2 = st.selectbox("Select Time 2", timestamps)

if t1 == t2:
    st.warning("Select two different timestamps for comparison.")
    st.stop()

# ---------------------
# FILTER DATA
# ---------------------
df1 = df[df["timestamp_IST"] == t1][["strike_price", "call_oi"]].rename(columns={"call_oi": "call_oi_t1"})
df2 = df[df["timestamp_IST"] == t2][["strike_price", "call_oi"]].rename(columns={"call_oi": "call_oi_t2"})

# ---------------------
# MERGE & CALCULATE CHANGE
# ---------------------
merged = pd.merge(df1, df2, on="strike_price", how="outer").sort_values("strike_price")

merged["OI_Change"] = merged["call_oi_t2"] - merged["call_oi_t1"]

# ---------------------
# COLOR FUNCTION
# ---------------------
def highlight_change(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "background-color: lightgreen"
    if val < 0:
        return "background-color: pink"
    return ""

# ---------------------
# SHOW RESULT
# ---------------------
st.subheader(f"Call OI Change from {t1} → {t2}")

st.dataframe(
    merged.style.applymap(highlight_change, subset=["OI_Change"])
)
