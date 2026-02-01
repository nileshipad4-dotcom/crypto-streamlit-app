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
DELTA_API = "https://api.india.delta.exchange/v2/tickers"

st.set_page_config(layout="wide", page_title="OI Time Scanner")

# -----------------------------------
# PRICE (DELTA — SAME AS YOUR APP)
# -----------------------------------
@st.cache_data(ttl=10)
def get_delta_price(symbol):
    r = requests.get(DELTA_API, timeout=10).json()["result"]
    return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])


# -----------------------------------
# HELPERS
# -----------------------------------
def get_available_expiries():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    expiries = sorted(
        list(set(f.split("_")[1].replace(".csv", "") for f in files)),
        key=lambda x: datetime.strptime(x, "%d-%m-%Y")
    )
    return expiries


def load_data(symbol, expiry):
    df = pd.read_csv(f"{DATA_DIR}/{symbol}_{expiry}.csv")
    df["_row"] = range(len(df))
    df["timestamp_IST"] = pd.to_datetime(df["timestamp_IST"], format="%H:%M")
    return df


def build_windows(df):
    times = (
        df.sort_values("_row")
          .drop_duplicates("timestamp_IST")["timestamp_IST"]
          .tolist()
    )

    windows = []
    i = 0
    while i < len(times) - 1:
        t1 = times[i]
        target = t1 + timedelta(minutes=MIN_GAP_MINUTES)

        t2 = None
        for t in times[i + 1:]:
            if t >= target:
                t2 = t
                break

        if t2 is None:
            break

        windows.append((t1, t2))
        i = times.index(t2)

    return windows


def process(df):
    rows = []

    for t1, t2 in build_windows(df):
        d1 = df[df["timestamp_IST"] == t1]
        d2 = df[df["timestamp_IST"] == t2]

        m = pd.merge(d1, d2, on="strike_price", suffixes=("_1", "_2"))
        m["CE"] = m["call_oi_2"] - m["call_oi_1"]
        m["PE"] = m["put_oi_2"] - m["put_oi_1"]

        ce = m.sort_values("CE", ascending=False)
        pe = m.sort_values("PE", ascending=False)

        rows.append({
            "TIME": f"{t1:%H:%M} - {t2:%H:%M}",

            "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
            "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
            "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
            "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",

            "_ce1": ce.iloc[0].CE,
            "_ce2": ce.iloc[1].CE,
            "_pe1": pe.iloc[0].PE,
            "_pe2": pe.iloc[1].PE,
        })

    return pd.DataFrame(rows)


# -----------------------------------
# COLUMN-WISE HIGHLIGHTING (CORRECT)
# -----------------------------------
def highlight(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for col, num_col in [
        ("MAX CE 1", "_ce1"),
        ("MAX CE 2", "_ce2"),
        ("MAX PE 1", "_pe1"),
        ("MAX PE 2", "_pe2"),
    ]:
        vals = df[num_col].abs()
        top1, top2 = vals.nlargest(2).values

        for i, v in vals.items():
            if v == top1:
                styles.loc[i, col] = "background-color:#ff4d4d;color:white;font-weight:bold"
            elif v == top2:
                styles.loc[i, col] = "background-color:#ffa500;font-weight:bold"

    return df[["TIME","MAX CE 1","MAX CE 2","MAX PE 1","MAX PE 2"]].style.apply(lambda _: styles, axis=None)


# -----------------------------------
# UI
# -----------------------------------
st.title("BTC & ETH — OI Time Window Scanner")

# Prices (Delta)
p1, p2 = st.columns(2)
p1.metric("BTC Price", f"{int(get_delta_price('BTC')):,}")
p2.metric("ETH Price", f"{int(get_delta_price('ETH')):,}")

expiry = st.selectbox("Select Expiry", get_available_expiries(), index=-1)
st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("BTC")
    df = process(load_data("BTC", expiry))
    st.dataframe(highlight(df), use_container_width=True)

with c2:
    st.subheader("ETH")
    df = process(load_data("ETH", expiry))
    st.dataframe(highlight(df), use_container_width=True)
