import streamlit as st
import pandas as pd
import requests
from datetime import timedelta, datetime
import os
from streamlit_autorefresh import st_autorefresh
import base64
from io import StringIO

# -----------------------------------
# CONFIG
# -----------------------------------
DATA_DIR = "data"
MIN_GAP_MINUTES = 15
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

st.set_page_config(layout="wide", page_title="OI Time Scanner")

# Auto refresh every 5 seconds
st_autorefresh(interval=200_000, key="price_refresh")

if "last_push_ts" not in st.session_state:
    st.session_state.last_push_ts = None

# -----------------------------------
# HELPERS
# -----------------------------------

DELTA_API = "https://api.india.delta.exchange/v2/tickers"

# -----------------------------------
# GITHUB CONFIG
# -----------------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
CRYPTO_REPO = "nileshipad4-dotcom/crypto-streamlit-app"
GITHUB_BRANCH = "main"
GITHUB_API = "https://api.github.com"


@st.cache_data(ttl=5)
def get_delta_price(symbol):
    """
    Fetch Delta Exchange USD perpetual mark price
    symbol: 'BTC' or 'ETH'
    """
    try:
        r = requests.get(DELTA_API, timeout=5).json()["result"]
        return float(
            next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"]
        )
    except Exception:
        return None

def get_available_expiries():
    expiries = set()

    for f in os.listdir(DATA_DIR):
        if not f.endswith(".csv"):
            continue

        parts = f.replace(".csv", "").split("_")

        # Expect format: BTC_25-10-2024.csv
        if len(parts) != 2:
            continue

        expiry = parts[1]

        try:
            datetime.strptime(expiry, "%d-%m-%Y")
            expiries.add(expiry)
        except ValueError:
            # skips BTC_OI_WINDOWS.csv etc
            continue

    return sorted(expiries, key=lambda x: datetime.strptime(x, "%d-%m-%Y"))



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


def build_row(df, t1, t2, is_live=False):
    d1 = df[df["timestamp_IST"] == t1]
    d2 = df[df["timestamp_IST"] == t2]

    if d1.empty or d2.empty:
        return None

    m = pd.merge(d1, d2, on="strike_price", suffixes=("_1", "_2"))

    strikes = sorted(m["strike_price"].unique())
    if len(strikes) <= 4:
        return None
    m = m[m["strike_price"].isin(strikes[2:-2])]

    m["CE"] = m["call_oi_2"] - m["call_oi_1"]
    m["PE"] = m["put_oi_2"] - m["put_oi_1"]

    ce = m.sort_values("CE", ascending=False)
    pe = m.sort_values("PE", ascending=False)

    if len(ce) < 2 or len(pe) < 2:
        return None

    sum_ce = int(m["CE"].sum() / 100)
    sum_pe = int(m["PE"].sum() / 100)
    diff = sum_ce - sum_pe

    label = f"{t1:%H:%M} - {t2:%H:%M}"
    if is_live:
        label += " ⏳"

    return {
        "TIME": label,
        "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
        "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
        "Σ ΔCE OI": sum_ce,
        "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
        "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",
        "Σ ΔPE OI": sum_pe,
        "Δ (CE − PE)": diff,
        "_ce1": ce.iloc[0].CE,
        "_ce2": ce.iloc[1].CE,
        "_pe1": pe.iloc[0].PE,
        "_pe2": pe.iloc[1].PE,
        "_is_live": is_live,
    }

def process_windows(df):
    rows = []

    windows = build_all_windows(df)

    # ---- FIXED WINDOWS ----
    for t1, t2 in windows:
        row = build_row(df, t1, t2, is_live=False)
        if row:
            rows.append(row)

    # ---- LIVE WINDOW (rolling) ----
    if windows:
        live_start = windows[-1][1]   # last fixed end
    else:
        live_start = df["timestamp_IST"].min()

    live_end = df["timestamp_IST"].max()

    if live_end > live_start:
        row = build_row(df, live_start, live_end, is_live=True)
        if row:
            rows.append(row)

    return pd.DataFrame(rows)



def append_csv_to_github(path, new_df, commit_msg):
    url = f"{GITHUB_API}/repos/{CRYPTO_REPO}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        content = base64.b64decode(r.json()["content"]).decode()
        sha = r.json()["sha"]
        old_df = pd.read_csv(StringIO(content))
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df
        sha = None

    csv_text = final_df.to_csv(index=False)

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(csv_text.encode()).decode(),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    requests.put(url, headers=headers, json=payload)


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

btc_price = get_delta_price("BTC")
eth_price = get_delta_price("ETH")


p1, p2 = st.columns(2)

p1.metric(
    "BTC Price (Delta USD Perp)",
    f"{btc_price:,.2f}" if btc_price else "—"
)

p2.metric(
    "ETH Price (Delta USD Perp)",
    f"{eth_price:,.2f}" if eth_price else "—"
)


expiries = get_available_expiries()
expiry = st.selectbox("Select Expiry", expiries, index=len(expiries) - 1)

push_to_github = st.toggle("📤 Push OI window data to GitHub CSV", value=False)


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

# -----------------------------------
# PUSH RAW SNAPSHOT ONLY (COLLECTOR STYLE)
# -----------------------------------

def get_ist_hhmm():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

@st.cache_data(ttl=15)
def fetch_live_collector_data(underlying, expiry):
    url = (
        f"{DELTA_API}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry}"
    )

    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame()

    raw = r.json().get("result", [])
    if not raw:
        return pd.DataFrame()

    df = pd.json_normalize(raw)

    df = df[
        [
            "strike_price", "contract_type",
            "mark_price", "oi_contracts", "volume",
            "greeks.gamma", "greeks.delta", "greeks.vega"
        ]
    ]

    df.rename(columns={
        "greeks.gamma": "gamma",
        "greeks.delta": "delta",
        "greeks.vega": "vega",
    }, inplace=True)

    for c in ["strike_price", "mark_price", "oi_contracts", "volume", "gamma", "delta", "vega"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    calls = df[df["contract_type"] == "call_options"].copy()
    puts  = df[df["contract_type"] == "put_options"].copy()

    calls = calls.rename(columns={
        "mark_price": "call_mark",
        "oi_contracts": "call_oi",
        "volume": "call_volume",
        "gamma": "call_gamma",
        "delta": "call_delta",
        "vega": "call_vega",
    }).drop(columns=["contract_type"])

    puts = puts.rename(columns={
        "mark_price": "put_mark",
        "oi_contracts": "put_oi",
        "volume": "put_volume",
        "gamma": "put_gamma",
        "delta": "put_delta",
        "vega": "put_vega",
    }).drop(columns=["contract_type"])

    merged = pd.merge(calls, puts, on="strike_price", how="outer")
    merged = merged.sort_values("strike_price").reset_index(drop=True)

    merged["Expiry"] = expiry
    merged["timestamp_IST"] = get_ist_hhmm()

    return merged


def compute_max_pain_collector(df):
    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    mp = []
    for i in range(len(df)):
        q = -sum(A[i:] * B[i:])
        r = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        s = -sum(M[:i] * L[:i])
        t = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        mp.append(round((q + r + s + t) / 10000))

    df["max_pain"] = mp
    return df


CANONICAL_COLS = [
    "call_mark","call_oi","call_volume",
    "call_gamma","call_delta","call_vega",
    "strike_price",
    "put_gamma","put_delta","put_vega",
    "put_volume","put_oi","put_mark",
    "Expiry","timestamp_IST","max_pain"
]


now_ts = get_ist_hhmm()

if push_to_github and st.session_state.last_push_ts != now_ts:

    update_msgs = []

    if not btc.empty:
        btc_out = btc.drop(columns=["_ce1","_ce2","_pe1","_pe2"], errors="ignore")
        btc_out["timestamp_IST"] = now_ts

        append_csv_to_github(
            path=f"data/BTC_{expiry}.csv",
            new_df=btc_out,
            commit_msg=f"BTC OI window @ {now_ts} IST"
        )

    if not eth.empty:
        eth_out = eth.drop(columns=["_ce1","_ce2","_pe1","_pe2"], errors="ignore")
        eth_out["timestamp_IST"] = now_ts

        append_csv_to_github(
            path=f"data/ETH_{expiry}.csv",
            new_df=eth_out,
            commit_msg=f"ETH OI window @ {now_ts} IST"
        )

    st.session_state.last_push_ts = now_ts
def get_ist_hhmm():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")

now_ts = get_ist_hhmm()

if push_to_github and st.session_state.last_push_ts != now_ts:

    update_msgs = []

    for underlying in ["BTC", "ETH"]:

        # 🔹 FETCH RAW OPTION CHAIN (PER STRIKE)
        df_live = fetch_live_collector_data(underlying, expiry)

        if df_live.empty:
            continue

        # 🔹 COMPUTE MAX PAIN
        df_live = compute_max_pain_collector(df_live)

        # 🔹 FORCE EXACT COLUMN ORDER
        df_live = df_live.reindex(columns=CANONICAL_COLS)

        # 🔹 APPEND (NEVER OVERWRITE)
        append_csv_to_github(
            path=f"data/{underlying}_{expiry}.csv",
            new_df=df_live,
            commit_msg=f"{underlying} snapshot @ {now_ts} IST"
        )

        update_msgs.append(
            f"✅ {underlying} collector snapshot pushed @ {now_ts} IST"
        )

    st.session_state.last_push_ts = now_ts

    for msg in update_msgs:
        st.success(msg)
