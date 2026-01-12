# crypto compare — FINAL COMPLETE VERSION (EXPIRY + MP + PCR + OI/VOLUME DELTAS)

import streamlit as st
import pandas as pd
import requests
import calendar
import os
from datetime import datetime, timedelta, date
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

# -------------------------------------------------
# FORCE FRESH RUN TOKEN
# -------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# -------------------------------------------------
# AUTO REFRESH (60s)
# -------------------------------------------------
if st_autorefresh(interval=60_000, key="auto_refresh"):
    st.cache_data.clear()
    st.session_state.run_id += 1

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
BASE_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "refs/heads/main/data/"
)

API_BASE = "https://api.india.delta.exchange/v2/tickers"
ASSETS = ["BTC", "ETH"]
PIVOT_TIME = "17:30"

STRIKE_COL_IDX = 6
TIMESTAMP_COL_IDX = 14
VALUE_COL_IDX = 19

CALL_OI_COL_IDX = 1
CALL_VOL_COL_IDX = 2
PUT_OI_COL_IDX = 11
PUT_VOL_COL_IDX = 10

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_ist_hhmm():
    return get_ist_datetime().strftime("%H:%M")

def safe_ratio(a, b):
    try:
        if b is None or pd.isna(b) or float(b) == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None

def extract_fresh_timestamps_from_github(asset, pivot=PIVOT_TIME):
    df = pd.read_csv(f"{BASE_RAW_URL}{asset}.csv")
    times = df.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5].dropna().unique().tolist()
    pivot_m = int(pivot[:2]) * 60 + int(pivot[3:])
    return sorted(times, key=lambda t: (pivot_m - (int(t[:2])*60 + int(t[3:]))) % 1440)

# -------------------------------------------------
# EXPIRY LOGIC (UNCHANGED)
# -------------------------------------------------
def get_expiries():
    ist_now = get_ist_datetime()
    today = ist_now.date()

    if ist_now.time() <= datetime.strptime("17:30", "%H:%M").time():
        latest_valid = today
    else:
        latest_valid = today + timedelta(days=1)

    expiries = set()
    expiries.add(latest_valid)

    d = today
    while d.weekday() != calendar.FRIDAY:
        d += timedelta(days=1)
    if d >= latest_valid:
        expiries.add(d)

    year, month = today.year, today.month
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[calendar.FRIDAY] != 0:
            fd = date(year, month, week[calendar.FRIDAY])
            if fd >= latest_valid:
                expiries.add(fd)

    for i in [1, 2]:
        m = month + i
        y = year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        cal = calendar.monthcalendar(y, m)
        lf = max(week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0)
        expiries.add(date(y, m, lf))

    return sorted(
        [d.strftime("%d-%m-%Y") for d in expiries],
        key=lambda x: datetime.strptime(x, "%d-%m-%Y"),
    )

# -------------------------------------------------
# CONTROLS
# -------------------------------------------------
if "ts_asset" not in st.session_state:
    st.session_state.ts_asset = "ETH"

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

c1, c2, c3 = st.columns([1, 6, 3])
with c1:
    st.selectbox("", ASSETS, key="ts_asset", label_visibility="collapsed")
with c2:
    if st.button("⏱"):
        st.session_state.timestamps = extract_fresh_timestamps_from_github(
            st.session_state.ts_asset
        )
with c3:
    expiry_list = get_expiries()
    selected_expiry = st.selectbox("Expiry", expiry_list)

if not st.session_state.timestamps:
    st.session_state.timestamps = extract_fresh_timestamps_from_github(
        st.session_state.ts_asset
    )

timestamps = st.session_state.timestamps
t1 = st.selectbox("Time 1 (Latest)", timestamps, index=0)
t2 = st.selectbox("Time 2 (Previous)", timestamps, index=1)

# -------------------------------------------------
# LIVE PRICES
# -------------------------------------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    r = requests.get(API_BASE, timeout=10).json()["result"]
    return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])

prices = {a: get_delta_price(a) for a in ASSETS}

p1, p2 = st.columns(2)
p1.metric("BTC Price", f"{int(prices['BTC']):,}")
p2.metric("ETH Price", f"{int(prices['ETH']):,}")

# -------------------------------------------------
# PCR COLLECTION
# -------------------------------------------------
pcr_rows = []

# =================================================
# MAIN LOOP
# =================================================
for UNDERLYING in ASSETS:

    file_path = f"data/{UNDERLYING}_{selected_expiry}.csv"

    if not os.path.exists(file_path):
        st.warning(f"No data file found for {UNDERLYING} {selected_expiry}")
        continue

    df_raw = pd.read_csv(file_path)


    df = pd.DataFrame({
        "strike_price": pd.to_numeric(df_raw.iloc[:, STRIKE_COL_IDX], errors="coerce"),
        "value": pd.to_numeric(df_raw.iloc[:, VALUE_COL_IDX], errors="coerce"),
        "call_oi": pd.to_numeric(df_raw.iloc[:, CALL_OI_COL_IDX], errors="coerce"),
        "put_oi": pd.to_numeric(df_raw.iloc[:, PUT_OI_COL_IDX], errors="coerce"),
        "call_vol": pd.to_numeric(df_raw.iloc[:, CALL_VOL_COL_IDX], errors="coerce"),
        "put_vol": pd.to_numeric(df_raw.iloc[:, PUT_VOL_COL_IDX], errors="coerce"),
        "timestamp": df_raw.iloc[:, TIMESTAMP_COL_IDX].astype(str).str[:5],
    }).dropna()

    # ---------- MP SNAPSHOTS ----------
    df_t1 = df[df["timestamp"] == t1].groupby("strike_price")["value"].sum()
    df_t2 = df[df["timestamp"] == t2].groupby("strike_price")["value"].sum()

    merged = pd.concat([df_t1, df_t2], axis=1)
    merged.columns = [f"MP ({t1})", f"MP ({t2})"]
    merged["△ MP 2"] = merged.iloc[:, 0] - merged.iloc[:, 1]

    # ---------- LIVE MP ----------
    df_live = pd.json_normalize(
        requests.get(
            f"{API_BASE}?contract_types=call_options,put_options"
            f"&underlying_asset_symbols={UNDERLYING}"
            f"&expiry_date={selected_expiry}",
            timeout=20,
        ).json()["result"]
    )

    # ---------- LIVE OI & VOLUME (REAL-TIME) ----------
    # ---------- LIVE OI & VOLUME (REAL-TIME) ----------
    
    df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")
    df_live["volume"] = pd.to_numeric(df_live["volume"], errors="coerce")
    
    live_agg = (
        df_live
        .groupby(["strike_price", "contract_type"])
        .agg({
            "oi_contracts": "sum",
            "volume": "sum"
        })
        .unstack()
        .fillna(0)
    )
    
    live_agg.columns = ["Call OI", "Put OI", "Call Volume", "Put Volume"]


    state_key = f"prev_live_{UNDERLYING}"
    
    if state_key not in st.session_state:
        st.session_state[state_key] = live_agg.copy()
    
    # Align both DataFrames to avoid shape mismatch
    prev = st.session_state[state_key]
    live_agg, prev = live_agg.align(prev, fill_value=0)
    
    delta_live = live_agg - prev
    
    delta_live = delta_live.rename(columns={
        "Call OI": "Δ Call OI",
        "Put OI": "Δ Put OI",
        "Call Volume": "Δ Call Volume",
        "Put Volume": "Δ Put Volume",
    })
    
    # Update snapshot for next refresh
    st.session_state[state_key] = live_agg.copy()



    df_mp = df_live[["strike_price", "contract_type", "mark_price", "oi_contracts"]].copy()
    for c in ["strike_price", "mark_price", "oi_contracts"]:
        df_mp[c] = pd.to_numeric(df_mp[c], errors="coerce")

    calls = df_mp[df_mp["contract_type"] == "call_options"]
    puts = df_mp[df_mp["contract_type"] == "put_options"]

    mp = pd.merge(
        calls.rename(columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}),
        puts.rename(columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}),
        on="strike_price",
        how="outer",
    ).sort_values("strike_price")

    def compute_max_pain(df):
        A, B = df["call_mark"].fillna(0), df["call_oi"].fillna(0)
        G = df["strike_price"]
        L, M = df["put_oi"].fillna(0), df["put_mark"].fillna(0)
        df["Current"] = [
            (
                -sum(A[i:] * B[i:]) + G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
                - sum(M[:i] * L[:i]) + sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
            ) / 10000
            for i in range(len(df))
        ]
        return df[["strike_price", "Current"]]

    live_mp = compute_max_pain(mp)
    now_ts = get_ist_hhmm()
    live_mp.columns = ["strike_price", f"MP ({now_ts})"]
  
    # ---------- OI & VOLUME DELTAS (LATEST CSV - T1) ----------
    
    # Use latest available CSV timestamp (not system time)
    current_time = df["timestamp"].max()
    
    current_df = df[df["timestamp"] == current_time]
    
    agg_current = current_df.groupby("strike_price")[
        ["call_oi", "put_oi", "call_vol", "put_vol"]
    ].sum()
    
    agg_t1 = df[df["timestamp"] == t1].groupby("strike_price")[
        ["call_oi", "put_oi", "call_vol", "put_vol"]
    ].sum()
    
    delta_oi_vol = (agg_current - agg_t1).rename(columns={
        "call_oi": "Δ Call OI",
        "put_oi": "Δ Put OI",
        "call_vol": "Δ Call Volume",
        "put_vol": "Δ Put Volume",
    })

    # ---------- FINAL MERGE (NOTHING REMOVED) ----------
    final = (
        merged
        .merge(live_mp, on="strike_price", how="left")
        .merge(delta_live, on="strike_price", how="left")
    )

    final["△ MP 1"] = final[f"MP ({now_ts})"] - final[f"MP ({t1})"]
    final["ΔΔ MP 1"] = -1 * (final["△ MP 1"].shift(-1) - final["△ MP 1"])

    final = final[
        [
            "strike_price",
            f"MP ({now_ts})",
            f"MP ({t1})",
            f"MP ({t2})",
            "△ MP 1",
            "△ MP 2",
            "ΔΔ MP 1",
            "Δ Call OI",
            "Δ Put OI",
            "Δ Call Volume",
            "Δ Put Volume",
        ]
    ].round(0).astype("Int64").reset_index(drop=True)

    # ---------- HIGHLIGHTING (UNCHANGED) ----------
    atm = prices[UNDERLYING]
    strikes = final["strike_price"].astype(float).tolist()
    atm_low = max([s for s in strikes if s <= atm], default=None)
    atm_high = min([s for s in strikes if s >= atm], default=None)
    min_mp = final[f"MP ({now_ts})"].min()

    def highlight(row):
        if row["strike_price"] in (atm_low, atm_high):
            return ["background-color:#4B0082"] * len(row)
        if row[f"MP ({now_ts})"] == min_mp:
            return ["background-color:#8B0000;color:white"] * len(row)
        return [""] * len(row)

    st.subheader(f"{UNDERLYING} — {t1} vs {t2}")
    st.dataframe(final.style.apply(highlight, axis=1), use_container_width=True)

    # ---------- PCR ----------
    pcr_rows.append([
        UNDERLYING,
        safe_ratio(
            df_live[df_live["contract_type"]=="put_options"]["oi_contracts"].sum(),
            df_live[df_live["contract_type"]=="call_options"]["oi_contracts"].sum(),
        ),
        safe_ratio(
            df[df["timestamp"]==t1]["put_oi"].sum(),
            df[df["timestamp"]==t1]["call_oi"].sum(),
        ),
        safe_ratio(
            df[df["timestamp"]==t2]["put_oi"].sum(),
            df[df["timestamp"]==t2]["call_oi"].sum(),
        ),
    ])

# -------------------------------------------------
# PCR TABLE
# -------------------------------------------------
pcr_df = pd.DataFrame(
    pcr_rows,
    columns=["Asset", "PCR OI (Current)", "PCR OI (T1)", "PCR OI (T2)"]
).set_index("Asset")

st.subheader("📊 PCR Snapshot")
st.dataframe(pcr_df.round(3), use_container_width=True)
