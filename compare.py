# compare_oi.py
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Call OI Change Comparator", layout="wide")

st.title("📊 Call OI Change — Strike-wise Comparison")

DATA_DIR = "data"  # adjust if your CSVs live elsewhere

# Choose underlying
underlying = st.selectbox("Select Underlying", ["BTC", "ETH"])

file_path = os.path.join(DATA_DIR, f"{underlying}.csv")

if not os.path.exists(file_path):
    st.error(f"No data file found: {file_path}. Run the collector or check the path.")
    st.stop()

@st.cache_data(ttl=60)
def load_df(path):
    df = pd.read_csv(path)
    # ensure columns exist and types
    if "strike_price" in df.columns:
        df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    if "call_oi" in df.columns:
        df["call_oi"] = pd.to_numeric(df["call_oi"], errors="coerce")
    # keep only rows that have a timestamp
    if "timestamp_IST" in df.columns:
        df = df[df["timestamp_IST"].notna()]
    return df

df = load_df(file_path)

timestamps = sorted(df["timestamp_IST"].unique())
if len(timestamps) < 2:
    st.warning("Not enough timestamps in data to compare. Wait for more collector runs.")
    st.dataframe(df.head(10))
    st.stop()

# select times
col1, col2 = st.columns(2)
with col1:
    t1 = st.selectbox("Time 1 (older)", timestamps, index=max(0, len(timestamps)-2))
with col2:
    t2 = st.selectbox("Time 2 (newer)", timestamps, index=len(timestamps)-1)

if t1 == t2:
    st.warning("Please select two different timestamps.")
    st.stop()

# filter and prepare
df1 = df[df["timestamp_IST"] == t1][["strike_price", "call_oi"]].rename(columns={"call_oi": f"call_oi_{t1}"})
df2 = df[df["timestamp_IST"] == t2][["strike_price", "call_oi"]].rename(columns={"call_oi": f"call_oi_{t2}"})

merged = pd.merge(df1, df2, on="strike_price", how="outer").sort_values("strike_price").reset_index(drop=True)
merged[f"OI_Change_{t2}_minus_{t1}"] = merged[f"call_oi_{t2}"] - merged[f"call_oi_{t1}"]

# present results
st.subheader(f"Call OI change: {underlying} — {t1} → {t2}")

# add sorting / filters
sort_by = st.selectbox("Sort by", [f"OI_Change_{t2}_minus_{t1}", "strike_price"], index=0)
ascending = st.checkbox("Ascending", value=False)

display_df = merged.copy()
display_df = display_df.sort_values(by=sort_by, ascending=ascending)

# color styling
def style_change(v):
    if pd.isna(v):
        return ""
    if v > 0:
        return "background-color: #d4f4dd"  # light green
    if v < 0:
        return "background-color: #ffd8d8"  # light red
    return ""

st.dataframe(display_df.style.applymap(style_change, subset=[f"OI_Change_{t2}_minus_{t1}"]), use_container_width=True)

# totals summary
total_t1 = merged[f"call_oi_{t1}"].sum(min_count=1)
total_t2 = merged[f"call_oi_{t2}"].sum(min_count=1)
total_change = total_t2 - total_t1

st.markdown("---")
st.write(f"Total Call OI at {t1}: {total_t1:.0f}  •  at {t2}: {total_t2:.0f}  •  Change: {total_change:.0f}")
