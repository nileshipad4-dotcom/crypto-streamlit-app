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
    df = pd.read_csv(f"{DATA_DIR}/{symbol}_{expiry}.csv")
    df["_row"] = range(len(df))

    # ---- handle midnight rollover ----
    raw_times = pd.to_datetime(df["timestamp_IST"], format="%H:%M")
    base = datetime(2000, 1, 1)
    out, last, day = [], None, 0

    for t in raw_times:
        if last is not None and t < last:
            day += 1
        out.append(base + timedelta(days=day, hours=t.hour, minutes=t.minute))
        last = t

    df["timestamp_IST"] = out
    return df


def build_all_windows(df):
    times = (
        df.sort_values("_row")
          .drop_duplicates("timestamp_IST")["timestamp_IST"]
          .tolist()
    )

    windows, i = [], 0
    while i < len(times) - 1:
        t1 = times[i]
        target = t1 + timedelta(minutes=MIN_GAP_MINUTES)

        t2 = next((t for t in times[i+1:] if t >= target), None)
        if t2 is None:
            break

        windows.append((t1, t2))
        i = times.index(t2)

    return windows


def process_windows(df):
    rows = []

    for t1, t2 in build_all_windows(df):
        d1 = df[df["timestamp_IST"] == t1]
        d2 = df[df["timestamp_IST"] == t2]

        m = pd.merge(d1, d2, on="strike_price", suffixes=("_1", "_2"))

        # ---- remove top 2 & bottom 2 strikes ----
        strikes = sorted(m["strike_price"].unique())
        if len(strikes) <= 4:
            continue

        m = m[m["strike_price"].isin(strikes[2:-2])]

        # ---- deltas ----
        m["CE"] = m["call_oi_2"] - m["call_oi_1"]
        m["PE"] = m["put_oi_2"] - m["put_oi_1"]

        ce = m.sort_values("CE", ascending=False)
        pe = m.sort_values("PE", ascending=False)

        # ---- SUMS (new requirement) ----
        sum_ce = int(m["CE"].sum())
        sum_pe = int(m["PE"].sum())

        rows.append({
            "TIME": f"{t1:%H:%M} - {t2:%H:%M}",

            "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
            "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
            "Σ ΔCE OI": sum_ce,

            "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
            "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",
            "Σ ΔPE OI": sum_pe,

            "_ce1": ce.iloc[0].CE,
            "_ce2": ce.iloc[1].CE,
            "_pe1": pe.iloc[0].PE,
            "_pe2": pe.iloc[1].PE,
        })

    return pd.DataFrame(rows)


# -----------------------------------
# HIGHLIGHTING
# -----------------------------------
def highlight_table(df):
    display_cols = [
        "TIME",
        "MAX CE 1", "MAX CE 2", "Σ ΔCE OI",
        "MAX PE 1", "MAX PE 2", "Σ ΔPE OI"
    ]

    styles = pd.DataFrame("", index=df.index, columns=display_cols)

    # highlight CE / PE max columns
    for col, num_col in [
        ("MAX CE 1", "_ce1"),
        ("MAX CE 2", "_ce2"),
        ("MAX PE 1", "_pe1"),
        ("MAX PE 2", "_pe2"),
    ]:
        vals = df[num_col].abs()
        if len(vals) < 2:
            continue
        top1, top2 = vals.nlargest(2).values

        for i, v in vals.items():
            if v == top1:
                styles.loc[i, col] = "background-color:#ff4d4d;color:white;font-weight:bold"
            elif v == top2:
                styles.loc[i, col] = "background-color:#ffa500;font-weight:bold"

    # highlight SUM columns (directional)
    for col in ["Σ ΔCE OI", "Σ ΔPE OI"]:
        for i, v in df[col].items():
            if v > 0:
                styles.loc[i, col] = "background-color:#e6ffe6;font-weight:bold"
            elif v < 0:
                styles.loc[i, col] = "background-color:#ffe6e6;font-weight:bold"

    return df[display_cols].style.apply(lambda _: styles, axis=None)


# -----------------------------------
# UI
# -----------------------------------
st.title("BTC & ETH OI Change Scanner")

c1, c2 = st.columns(2)
c1.metric("BTC Price", get_binance_price("BTCUSDT"))
c2.metric("ETH Price", get_binance_price("ETHUSDT"))

expiries = get_available_expiries()
expiry = st.selectbox("Select Expiry", expiries, index=len(expiries) - 1)

st.divider()

b1, b2 = st.columns(2)

with b1:
    st.subheader("BTC")
    btc = process_windows(load_data("BTC", expiry))
    st.dataframe(highlight_table(btc), use_container_width=True)

with b2:
    st.subheader("ETH")
    eth = process_windows(load_data("ETH", expiry))
    st.dataframe(highlight_table(eth), use_container_width=True)
