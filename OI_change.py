import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

DATA_DIR = "data"
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

st.set_page_config(layout="wide", page_title="OI Scanner")

# ---------------------------------------
# Helpers
# ---------------------------------------
def get_binance_price(symbol):
    try:
        r = requests.get(BINANCE_API, params={"symbol": symbol}, timeout=5)
        return float(r.json()["price"])
    except:
        return None


def get_available_expiries():
    files = os.listdir(DATA_DIR)
    expiries = sorted(
        list(set(f.split("_")[1].replace(".csv", "") for f in files if f.endswith(".csv"))),
        key=lambda x: datetime.strptime(x, "%d-%m-%Y")
    )
    return expiries


def load_data(symbol, expiry):
    path = f"{DATA_DIR}/{symbol}_{expiry}.csv"
    df = pd.read_csv(path)
    df["timestamp_IST"] = pd.to_datetime(df["timestamp_IST"], format="%H:%M")
    return df


def find_time_window(times):
    base = times.min()
    target = base + timedelta(minutes=15)
    later_times = times[times >= target]
    if later_times.empty:
        return None, None
    return base, later_times.min()


def build_oi_table(df):
    t1, t2 = find_time_window(df["timestamp_IST"])
    if t1 is None:
        return None, None

    df1 = df[df["timestamp_IST"] == t1]
    df2 = df[df["timestamp_IST"] == t2]

    merged = pd.merge(
        df1, df2,
        on="strike_price",
        suffixes=("_t1", "_t2")
    )

    merged["call_oi_change"] = merged["call_oi_t2"] - merged["call_oi_t1"]
    merged["put_oi_change"] = merged["put_oi_t2"] - merged["put_oi_t1"]

    ce = merged.sort_values("call_oi_change", ascending=False)
    pe = merged.sort_values("put_oi_change", ascending=False)

    result = {
        "TIME": f"{t1.strftime('%H:%M')} - {t2.strftime('%H:%M')}",
        "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].call_oi_change)}",
        "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].call_oi_change)}",
        "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].put_oi_change)}",
        "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].put_oi_change)}",
    }

    numeric_values = {
        "MAX CE 1": ce.iloc[0].call_oi_change,
        "MAX CE 2": ce.iloc[1].call_oi_change,
        "MAX PE 1": pe.iloc[0].put_oi_change,
        "MAX PE 2": pe.iloc[1].put_oi_change,
    }

    return pd.DataFrame([result]), numeric_values


def highlight_cells(df, values):
    max1 = max(values.values())
    max2 = sorted(values.values(), reverse=True)[1]

    def style(val):
        for k, v in values.items():
            if v == max1 and str(v) in str(val):
                return "background-color: #ff4d4d; color: white; font-weight: bold"
            if v == max2 and str(v) in str(val):
                return "background-color: #ffa500; font-weight: bold"
        return ""

    return df.style.applymap(style)


# ---------------------------------------
# UI
# ---------------------------------------
st.title("BTC & ETH OI Change Scanner")

# Prices
col1, col2 = st.columns(2)
with col1:
    st.metric("BTC Price", get_binance_price("BTCUSDT"))
with col2:
    st.metric("ETH Price", get_binance_price("ETHUSDT"))

# Expiry selector
expiries = get_available_expiries()
default_idx = len(expiries) - 1
expiry = st.selectbox("Select Expiry Date", expiries, index=default_idx)

st.divider()

# Tables
col_btc, col_eth = st.columns(2)

with col_btc:
    st.subheader("BTC")
    btc_df = load_data("BTC", expiry)
    table, vals = build_oi_table(btc_df)
    if table is not None:
        st.dataframe(highlight_cells(table, vals), use_container_width=True)
    else:
        st.warning("Not enough data")

with col_eth:
    st.subheader("ETH")
    eth_df = load_data("ETH", expiry)
    table, vals = build_oi_table(eth_df)
    if table is not None:
        st.dataframe(highlight_cells(table, vals), use_container_width=True)
    else:
        st.warning("Not enough data")
