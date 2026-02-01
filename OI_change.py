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

    # ---- midnight rollover fix ----
    raw = pd.to_datetime(df["timestamp_IST"], format="%H:%M")
    base = datetime(2000, 1, 1)
    out, last, day = [], None, 0

    for t in raw:
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

        t2 = next((t for t in times[i + 1:] if t >= target), None)
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

        ce = (
            m.sort_values("CE", ascending=False)
             .drop_duplicates("strike_price")
        )
        pe = (
            m.sort_values("PE", ascending=False)
             .drop_duplicates("strike_price")
        )

        if len(ce) < 2 or len(pe) < 2:
            continue

        # ---- scaled sums ----
        sum_ce = int(m["CE"].sum() / 100)
        sum_pe = int(m["PE"].sum() / 100)
        diff_ce_pe = sum_ce - sum_pe

        rows.append({
            "TIME": f"{t1:%H:%M} - {t2:%H:%M}",

            "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
            "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
            "Σ ΔCE OI": sum_ce,

            "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
            "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",
            "Σ ΔPE OI": sum_pe,

            "Δ (CE − PE)": diff_ce_pe,

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
        "MAX PE 1", "MAX PE 2", "Σ ΔPE OI",
        "Δ (CE − PE)",
    ]

    styles = pd.DataFrame("", index=df.index, columns=display_cols)

    # Highlight CE / PE max columns
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

    # ---- SUM COLOR LOGIC (TEXT ONLY) ----
    for i in df.index:
        diff = df.loc[i, "Δ (CE − PE)"]
        if diff > 0:
            color = "red"
        elif diff < 0:
            color = "green"
        else:
            continue

        styles.loc[i, "Σ ΔCE OI"] = f"color:{color};font-weight:bold"
        styles.loc[i, "Σ ΔPE OI"] = f"color:{color};font-weight:bold"
        styles.loc[i, "Δ (CE − PE)"] = f"color:{color};font-weight:bold"

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

# -------- BTC TABLE --------
st.subheader("BTC")
btc = process_windows(load_data("BTC", expiry))
st.dataframe(highlight_table(btc), use_container_width=True)

# -------- BTC WINDOW DELTA (CUMULATIVE) --------
st.subheader("BTC — Window Delta Comparison (Cumulative)")

if not btc.empty:
    btc_times = btc["TIME"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        btc_from = st.selectbox("From Time (BTC)", btc_times, index=0, key="btc_from")
    with col2:
        btc_to = st.selectbox("To Time (BTC)", btc_times, index=len(btc_times)-1, key="btc_to")

    idx_from = btc_times.index(btc_from)
    idx_to = btc_times.index(btc_to)

    if idx_from <= idx_to:
        slice_df = btc.iloc[idx_from:idx_to + 1]
    else:
        slice_df = btc.iloc[idx_to:idx_from + 1]

    d_ce = int(slice_df["Σ ΔCE OI"].sum())
    d_pe = int(slice_df["Σ ΔPE OI"].sum())
    d_diff = int(slice_df["Δ (CE − PE)"].sum())

    def c(v):
        return "red" if v > 0 else "green" if v < 0 else "black"

    a, b, c3 = st.columns(3)
    a.markdown(f"**△ CE:** <span style='color:{c(d_ce)}'>{d_ce}</span>", unsafe_allow_html=True)
    b.markdown(f"**△ PE:** <span style='color:{c(d_pe)}'>{d_pe}</span>", unsafe_allow_html=True)
    c3.markdown(f"**△ (CE − PE):** <span style='color:{c(d_diff)}'>{d_diff}</span>", unsafe_allow_html=True)

st.divider()

# -------- ETH TABLE --------
st.subheader("ETH")
eth = process_windows(load_data("ETH", expiry))
st.dataframe(highlight_table(eth), use_container_width=True)

# -------- ETH WINDOW DELTA (CUMULATIVE) --------
st.subheader("ETH — Window Delta Comparison (Cumulative)")

if not eth.empty:
    eth_times = eth["TIME"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        eth_from = st.selectbox("From Time (ETH)", eth_times, index=0, key="eth_from")
    with col2:
        eth_to = st.selectbox("To Time (ETH)", eth_times, index=len(eth_times)-1, key="eth_to")

    idx_from = eth_times.index(eth_from)
    idx_to = eth_times.index(eth_to)

    if idx_from <= idx_to:
        slice_df = eth.iloc[idx_from:idx_to + 1]
    else:
        slice_df = eth.iloc[idx_to:idx_from + 1]

    d_ce = int(slice_df["Σ ΔCE OI"].sum())
    d_pe = int(slice_df["Σ ΔPE OI"].sum())
    d_diff = int(slice_df["Δ (CE − PE)"].sum())

    def c(v):
        return "red" if v > 0 else "green" if v < 0 else "black"

    a, b, c3 = st.columns(3)
    a.markdown(f"**△ CE:** <span style='color:{c(d_ce)}'>{d_ce}</span>", unsafe_allow_html=True)
    b.markdown(f"**△ PE:** <span style='color:{c(d_pe)}'>{d_pe}</span>", unsafe_allow_html=True)
    c3.markdown(f"**△ (CE − PE):** <span style='color:{c(d_diff)}'>{d_diff}</span>", unsafe_allow_html=True)
