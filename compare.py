import streamlit as st
import pandas as pd

st.title("📊 Max Pain (U Column) — 3-Way Comparison Tool")

# ------------------------------------------------
# 1️⃣ SELECT UNDERLYING
# ------------------------------------------------
underlying = st.selectbox("Select Underlying", ["BTC", "ETH"])

file_path = f"data/{underlying}.csv"
df = pd.read_csv(file_path)

# Ensure numeric types
for col in ["strike_price", "max_pain"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------
# 2️⃣ LOAD REAL-TIME DATA FROM app.py EXPORT
# (Assumes app.py provides df_live_maxpain externally)
# ------------------------------------------------
st.subheader("Upload Live Max Pain CSV from app.py")

live_file = st.file_uploader("Upload Live Snapshot (with strike_price + max_pain)", type=["csv"])

if live_file is None:
    st.info("Upload the real-time snapshot exported from app.py")
    st.stop()

df_live = pd.read_csv(live_file)
df_live["strike_price"] = pd.to_numeric(df_live["strike_price"], errors="coerce")
df_live["max_pain"] = pd.to_numeric(df_live["max_pain"], errors="coerce")
df_live = df_live[["strike_price", "max_pain"]].rename(columns={"max_pain": "live_mp"})

# ------------------------------------------------
# 3️⃣ SELECT TWO HISTORICAL TIMESTAMPS FROM CSV
# ------------------------------------------------
st.subheader("Select Historical Timeframes (from BTC.csv)")

timestamps = sorted(df["timestamp_IST"].unique())

t1 = st.selectbox("Select Historical Timestamp 1", timestamps)
t2 = st.selectbox("Select Historical Timestamp 2", timestamps)

if t1 == t2:
    st.warning("Select two different timestamps.")
    st.stop()

df_t1 = df[df["timestamp_IST"] == t1][["strike_price", "max_pain"]].rename(columns={"max_pain": "mp_t1"})
df_t2 = df[df["timestamp_IST"] == t2][["strike_price", "max_pain"]].rename(columns={"max_pain": "mp_t2"})

# ------------------------------------------------
# 4️⃣ MERGE ALL THREE DATASETS
# ------------------------------------------------
merged = pd.merge(df_live, df_t1, on="strike_price", how="outer")
merged = pd.merge(merged, df_t2, on="strike_price", how="outer")

# ------------------------------------------------
# 5️⃣ ADD COMPARISON COLUMNS
# ------------------------------------------------
merged["diff_live_t1"] = merged["live_mp"] - merged["mp_t1"]
merged["diff_t1_t2"] = merged["mp_t1"] - merged["mp_t2"]

# Order columns nicely
merged = merged.sort_values("strike_price")[[
    "strike_price",
    "live_mp",
    "mp_t1",
    "diff_live_t1",
    "mp_t2",
    "diff_t1_t2"
]]

# ------------------------------------------------
# 6️⃣ COLOR HIGHLIGHTS FOR DIFFERENCES
# ------------------------------------------------
def highlight_diff(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "background-color: lightgreen"
    if val < 0:
        return "background-color: pink"
    return ""

# ------------------------------------------------
# 7️⃣ SHOW TABLE
# ------------------------------------------------
st.subheader(f"3-Way Max Pain Comparison  —  Live vs {t1} vs {t2}")

st.dataframe(
    merged.style.applymap(highlight_diff, subset=["diff_live_t1", "diff_t1_t2"])
)

# DONE
