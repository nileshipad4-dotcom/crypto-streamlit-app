import streamlit as st
import pandas as pd

st.title("📊 3-Way Max Pain Comparison Tool")

# --------------------------------------------------------
# 1. CHOOSE UNDERLYING (BTC / ETH)
# --------------------------------------------------------
underlying = st.selectbox("Select Underlying", ["BTC", "ETH"])

file_path = f"data/{underlying}.csv"
df = pd.read_csv(file_path)

# Convert numeric columns
df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
df["call_mark"] = pd.to_numeric(df["call_mark"], errors="coerce")
df["call_oi"] = pd.to_numeric(df["call_oi"], errors="coerce")
df["put_mark"] = pd.to_numeric(df["put_mark"], errors="coerce")
df["put_oi"] = pd.to_numeric(df["put_oi"], errors="coerce")

# --------------------------------------------------------
# 2. FUNCTION TO COMPUTE MAX PAIN FOR ONE TIMESTAMP GROUP
# --------------------------------------------------------
def compute_mp_for_snapshot(df_snap):
    df_snap = df_snap.copy()

    A = df_snap["call_mark"].fillna(0).values
    B = df_snap["call_oi"].fillna(0).values
    G = df_snap["strike_price"].fillna(0).values
    L = df_snap["put_oi"].fillna(0).values
    M = df_snap["put_mark"].fillna(0).values

    n = len(df_snap)
    U = []

    for i in range(n):
        Q = -sum(A[i:] * B[i:])
        R = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        S = -sum(M[:i] * L[:i])
        T = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        U.append(round((Q + R + S + T) / 10000))

    df_snap["max_pain"] = U
    return df_snap[["strike_price", "max_pain"]]


# --------------------------------------------------------
# 3. COMPUTE MAX PAIN FOR ALL TIMESTAMPS
# --------------------------------------------------------
timestamps = sorted(df["timestamp_IST"].unique())

mp_dict = {}

for ts in timestamps:
    df_snap = df[df["timestamp_IST"] == ts]
    mp_dict[ts] = compute_mp_for_snapshot(df_snap)


# --------------------------------------------------------
# 4. REAL-TIME APP DATA (U column)
# --------------------------------------------------------
st.subheader("Upload Live Max Pain CSV (from app.py)")

live_file = st.file_uploader("Upload real-time snapshot", type=["csv"])

if live_file is None:
    st.info("Upload live snapshot exported from app.py")
    st.stop()

df_live = pd.read_csv(live_file)
df_live["strike_price"] = pd.to_numeric(df_live["strike_price"], errors="coerce")
df_live["max_pain"] = pd.to_numeric(df_live["max_pain"], errors="coerce")
df_live = df_live[["strike_price", "max_pain"]].rename(columns={"max_pain": "live_mp"})

# --------------------------------------------------------
# 5. SELECT 2 HISTORICAL TIMESTAMPS
# --------------------------------------------------------
st.subheader("Select 2 Historical Timeframes")

t1 = st.selectbox("Timestamp 1", timestamps)
t2 = st.selectbox("Timestamp 2", timestamps)

if t1 == t2:
    st.warning("Select two different timestamps.")
    st.stop()

df_t1 = mp_dict[t1].rename(columns={"max_pain": "mp_t1"})
df_t2 = mp_dict[t2].rename(columns={"max_pain": "mp_t2"})

# --------------------------------------------------------
# 6. MERGE: LIVE + T1 + T2
# --------------------------------------------------------
merged = pd.merge(df_live, df_t1, on="strike_price", how="outer")
merged = pd.merge(merged, df_t2, on="strike_price", how="outer")

# --------------------------------------------------------
# 7. DIFFERENCE COLUMNS
# --------------------------------------------------------
merged["diff_live_t1"] = merged["live_mp"] - merged["mp_t1"]
merged["diff_t1_t2"] = merged["mp_t1"] - merged["mp_t2"]

# Order columns
merged = merged.sort_values("strike_price")[[
    "strike_price",
    "live_mp",
    "mp_t1",
    "diff_live_t1",
    "mp_t2",
    "diff_t1_t2"
]]


# --------------------------------------------------------
# 8. COLOR FUNCTION
# --------------------------------------------------------
def highlight_diff(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "background-color: lightgreen"
    if val < 0:
        return "background-color: pink"
    return ""

# --------------------------------------------------------
# 9. DISPLAY RESULT
# --------------------------------------------------------
st.subheader(f"Max Pain Comparison — Live vs {t1} vs {t2}")

st.dataframe(
    merged.style.applymap(highlight_diff, subset=["diff_live_t1", "diff_t1_t2"])
)
