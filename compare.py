import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison Between Two Times (BTC.csv)")

# -------------------------------------------------
# LOAD BTC.csv
# -------------------------------------------------
BTC_PATH = "data/BTC.csv"
df = pd.read_csv(BTC_PATH)

# -------------------------------------------------
# FIXED COLUMN INDICES (AS CONFIRMED)
# -------------------------------------------------
STRIKE_COL_IDX = 6     # G:G
VALUE_COL_IDX = 16     # Q:Q (column to compare)
TIMESTAMP_COL_IDX = 14 # O:O

# -------------------------------------------------
# EXTRACT REQUIRED DATA BY POSITION
# -------------------------------------------------
df_extracted = pd.DataFrame({
    "strike_price": pd.to_numeric(df.iloc[:, STRIKE_COL_IDX], errors="coerce"),
    "value": pd.to_numeric(df.iloc[:, VALUE_COL_IDX], errors="coerce"),
    "timestamp": df.iloc[:, TIMESTAMP_COL_IDX]
})

# Drop invalid rows
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
    .groupby("strike_price", as_index=False)["value"]
    .sum()
    .rename(columns={"value": "value_time_1"})
)

df_t2 = (
    df_extracted[df_extracted["timestamp"] == t2]
    .groupby("strike_price", as_index=False)["value"]
    .sum()
    .rename(columns={"value": "value_time_2"})
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

merged["change"] = merged["value_time_2"] - merged["value_time_1"]

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
st.subheader(f"Comparison: {t1} → {t2}")

st.dataframe(
    merged.style.applymap(color_change, subset=["change"]),
    use_container_width=True
)

st.caption("🟢 Increase | 🔴 Decrease")
