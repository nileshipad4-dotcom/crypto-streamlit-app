import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("📊 Call OI Comparison Between Two Times (BTC.csv)")

# -------------------------------------------------
# LOAD BTC.csv
# -------------------------------------------------
BTC_PATH = "data/BTC.csv"
df = pd.read_csv(BTC_PATH)

# -------------------------------------------------
# FIXED COLUMN LOCATIONS (AS CONFIRMED)
# -------------------------------------------------
STRIKE_COL_IDX = 6    # Column G
CALL_OI_COL_IDX = 1   # Column B
TIMESTAMP_COL = "timestamp_IST"

# -------------------------------------------------
# BASIC VALIDATION
# -------------------------------------------------
if TIMESTAMP_COL not in df.columns:
    st.error(f"Column '{TIMESTAMP_COL}' not found in BTC.csv")
    st.write("Available columns:", list(df.columns))
    st.stop()

# -------------------------------------------------
# EXTRACT REQUIRED DATA
# -------------------------------------------------
df_extracted = pd.DataFrame({
    "strike_price": pd.to_numeric(df.iloc[:, STRIKE_COL_IDX], errors="coerce"),
    "call_oi": pd.to_numeric(df.iloc[:, CALL_OI_COL_IDX], errors="coerce"),
    "timestamp": df[TIMESTAMP_COL]
})

# Drop rows with missing strike or timestamp
df_extracted = df_extracted.dropna(subset=["strike_price", "timestamp"])

# -------------------------------------------------
# TIME SELECTION
# -------------------------------------------------
timestamps = sorted(df_extracted["timestamp"].unique())

if len(timestamps) < 2:
    st.error("Not enough timestamps in BTC.csv to compare.")
    st.stop()

t1 = st.selectbox("Select Time 1", timestamps, index=0)
t2 = st.selectbox("Select Time 2", timestamps, index=1)

if t1 == t2:
    st.warning("Please select two different timestamps.")
    st.stop()

# -------------------------------------------------
# FILTER DATA FOR BOTH TIMES
# -------------------------------------------------
df_t1 = (
    df_extracted[df_extracted["timestamp"] == t1]
    .groupby("strike_price", as_index=False)["call_oi"]
    .sum()
    .rename(columns={"call_oi": "call_oi_time_1"})
)

df_t2 = (
    df_extracted[df_extracted["timestamp"] == t2]
    .groupby("strike_price", as_index=False)["call_oi"]
    .sum()
    .rename(columns={"call_oi": "call_oi_time_2"})
)

# -------------------------------------------------
# MERGE & COMPARE
# -------------------------------------------------
merged = pd.merge(
    df_t1,
    df_t2,
    on="strike_price",
    how="outer"
).sort_values("strike_price")

merged["call_oi_change"] = merged["call_oi_time_2"] - merged["call_oi_time_1"]

# -------------------------------------------------
# COLOR FUNCTION
# -------------------------------------------------
def color_change(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "background-color: lightgreen"
    if val < 0:
        return "background-color: lightcoral"
    return ""

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
st.subheader(f"Call OI Change: {t1} → {t2}")

st.dataframe(
    merged.style.applymap(color_change, subset=["call_oi_change"]),
    use_container_width=True
)

st.caption("🟢 Increase in Call OI | 🔴 Decrease in Call OI")

