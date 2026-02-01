import streamlit as st
import pandas as pd
import requests
from datetime import timedelta, datetime
import os

# -----------------------------------
# CONFIG
# -----------------------------------
DATA_DIR = "data"
MIN_GAP_MINUTES = 15
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

st.set_page_config(layout="wide", page_title="OI Time Scanner")

# -----------------------------------
# HELPERS
# -----------------------------------
def get_binance_price(symbol):
    try:
        r = requests.get(BINANCE_API, params={"symbol": symbol}, timeout=5)
        return float(r.json()["price"])
    except:
        return None


def get_available_expiries():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    expiries = sorted(
        list(set(f.split("_")[1].replace(".csv", "") for f in files)),
        key=lambda x: datetime.strptime(x, "%d-%m-%Y")
    )
    return expiries


def load_data(symbol, expiry):
    path = f"{DATA_DIR}/{symbol}_{expiry}.csv"
    df = pd.read_csv(path)
    df["timestamp_IST"] = pd.to_datetime(df["timestamp_IST"], format="%H:%M")
    return df


def build_all_windows(df):
    times = sorted(df["timestamp_IST"].unique())
    windows = []

    i = 0
    while i < len(times) - 1:
        t1 = times[i]
        target = t1 + timedelta(minutes=MIN_GAP_MINUTES)

        t2_candidates = [t for t in times if t >= target]
        if not t2_candidates:
            break

        t2 = t2_candidates[0]
        windows.append((t1, t2))

        i = times.index(t2)

    return windows


def process_windows(df):
    rows = []

    windows = build_all_windows(df)
    if not windows:
        return pd.DataFrame()

    for t1, t2 in windows:
        df1 = df[df["timestamp_IST"] == t1]
        df2 = df[df["timestamp_IST"] == t2]

        merged = pd.merge(
            df1, df2,
            on="strike_price",
            suffixes=("_t1", "_t2")
        )

        merged["CE_CHANGE"] = merged["call_oi_t2"] - merged["call_oi_t1"]
        merged["PE_CHANGE"] = merged["put_oi_t2"] - merged["put_oi_t1"]

        ce = merged.sort_values("CE_CHANGE", ascending=False)
        pe = merged.sort_values("PE_CHANGE", ascending=False)

        rows.append({
            "TIME": f"{t1.strftime('%H:%M')} - {t2.strftime('%H:%M')}",
            "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE_CHANGE)}",
            "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE_CHANGE)}",
            "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE_CHANGE)}",
            "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE_CHANGE)}",
            "_ce1": ce.iloc[0].CE_CHANGE,
            "_ce2": ce.iloc[1].CE_CHANGE,
            "_pe1": pe.iloc[0].PE_CHANGE,
            "_pe2": pe.iloc[1].PE_CHANGE,
        })

    df_out = pd.DataFrame(rows)
    return df_out


def highlight_table(df):
    numeric_cols = ["_ce1", "_ce2", "_pe1", "_pe2"]
    values = df[numeric_cols].abs().values.flatten()
    max1 = sorted(values, reverse=True)[0]
    max2 = sorted(values, reverse=True)[1]

    def style(val):
        try:
            num = int(val.split(":-")[1])
            if abs(num) == max1:
                return "background-color:#ff4d4d;color:white;font-weight:bold"
            if abs(num) == max2:
                return "background-color:#ffa500;font-weight:bold"
        except:
            pass
        return ""

    display_cols = ["TIME", "MAX CE 1", "MAX CE 2", "MAX PE 1", "MAX PE 2"]
    return df[display_cols].style.applymap(style)


# -----------------------------------
# UI
# -----------------------------------
st.title("BTC & ETH OI Change Scanner")

# Prices
c1, c2 = st.columns(2)
with c1:
    st.metric("BTC Price", get_binance_price("BTCUSDT"))
with c2:
    st.metric("ETH Price", get_binance_price("ETHUSDT"))

# Expiry selector
expiries = get_available_expiries()
expiry = st.selectbox("Select Expiry", expiries, index=len(expiries) - 1)

st.divider()

# Tables
btc_col, eth_col = st.columns(2)

with btc_col:
    st.subheader("BTC")
    btc_df = load_data("BTC", expiry)
    btc_table = process_windows(btc_df)
    if btc_table.empty:
        st.warning("Not enough data")
    else:
        st.dataframe(highlight_table(btc_table), use_container_width=True)

with eth_col:
    st.subheader("ETH")
    eth_df = load_data("ETH", expiry)
    eth_table = process_windows(eth_df)
    if eth_table.empty:
        st.warning("Not enough data")
    else:
        st.dataframe(highlight_table(eth_table), use_container_width=True)
