import streamlit as st
import pandas as pd
import requests
from datetime import timedelta, datetime
import os
from streamlit_autorefresh import st_autorefresh
import base64
from io import StringIO

# ===================================
# CONFIG
# ===================================

DATA_DIR = "data"
RAW_DIR = "data/raw"
MIN_GAP_MINUTES = 15

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="OI Time Scanner")
st_autorefresh(interval=60_000, key="price_refresh")

# ===================================
# SESSION STATE
# ===================================

if "last_push_min_bucket" not in st.session_state:
    st.session_state.last_push_min_bucket = None

# ===================================
# APIS
# ===================================

DELTA_API = "https://api.india.delta.exchange/v2/tickers"

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
CRYPTO_REPO = "nileshipad4-dotcom/crypto-streamlit-app"
GITHUB_BRANCH = "main"
GITHUB_API = "https://api.github.com"

# ===================================
# HELPERS
# ===================================

@st.cache_data(ttl=5)
def get_delta_price(symbol):
    try:
        r = requests.get(DELTA_API, timeout=5).json()["result"]
        return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])
    except Exception:
        return None


def get_available_expiries():
    expiries = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith(".csv"):
            try:
                expiry = f.split("_")[1].replace(".csv", "")
                datetime.strptime(expiry, "%d-%m-%Y")
                expiries.add(expiry)
            except Exception:
                pass
    return sorted(expiries, key=lambda x: datetime.strptime(x, "%d-%m-%Y"))


# ===================================
# CLEAN WINDOW DATA (READ ONLY)
# ===================================

def load_data(symbol, expiry):
    path = f"{DATA_DIR}/{symbol}_{expiry}.csv"
    df = pd.read_csv(path)
    df["_row"] = range(len(df))

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


# ===================================
# WINDOW LOGIC (UNCHANGED)
# ===================================

def build_all_windows(df, min_gap_minutes):
    times = (
        df.sort_values("_row")
        .drop_duplicates("timestamp_IST")["timestamp_IST"]
        .tolist()
    )

    windows, i = [], 0
    while i < len(times) - 1:
        t1 = times[i]
        target = t1 + timedelta(minutes=min_gap_minutes)
        t2 = next((t for t in times[i + 1:] if t >= target), None)
        if not t2:
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

    ce, pe = m.sort_values("CE", ascending=False), m.sort_values("PE", ascending=False)
    if len(ce) < 2 or len(pe) < 2:
        return None

    label = f"{t1:%H:%M} - {t2:%H:%M}" + (" ⏳" if is_live else "")
    return {
        "TIME": label,
        "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
        "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
        "Σ ΔCE OI": int(m["CE"].sum() / 100),
        "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
        "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",
        "Σ ΔPE OI": int(m["PE"].sum() / 100),
        "Δ (PE − CE)": int(m["PE"].sum() / 100) - int(m["CE"].sum() / 100),
        "_ce1": ce.iloc[0].CE,
        "_ce2": ce.iloc[1].CE,
        "_pe1": pe.iloc[0].PE,
        "_pe2": pe.iloc[1].PE,
    }


def process_windows(df):
    rows = []
    windows = build_all_windows(df, MIN_GAP_MINUTES)

    for t1, t2 in windows:
        r = build_row(df, t1, t2)
        if r:
            rows.append(r)

    if windows:
        r = build_row(df, windows[-1][1], df["timestamp_IST"].max(), is_live=True)
        if r:
            rows.append(r)

    return pd.DataFrame(rows)


# ===================================
# RAW COLLECTOR (SEPARATE FILE)
# ===================================

def get_ist_hhmm():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")


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

    df = pd.json_normalize(r.json().get("result", []))
    if df.empty:
        return df

    df = df[["strike_price", "contract_type", "mark_price", "oi_contracts"]]
    df["timestamp_IST"] = get_ist_hhmm()
    df["Expiry"] = expiry
    return df


def append_raw_snapshot(path, df):
    url = f"{GITHUB_API}/repos/{CRYPTO_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers)

    sha, old = None, ""
    if r.status_code == 200:
        sha = r.json()["sha"]
        old = base64.b64decode(r.json()["content"]).decode()

    csv = old + df.to_csv(index=False, header=not bool(old))
    payload = {
        "message": "raw snapshot",
        "content": base64.b64encode(csv.encode()).decode(),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    requests.put(url, headers=headers, json=payload)


# ===================================
# UI
# ===================================

st.title("BTC & ETH OI Change Scanner")

expiries = get_available_expiries()
expiry = st.selectbox("Select Expiry", expiries)

for sym in ["BTC", "ETH"]:
    st.subheader(sym)
    df = process_windows(load_data(sym, expiry))
    st.dataframe(df, use_container_width=True)

# ===================================
# COLLECTOR PUSH (RAW ONLY)
# ===================================

push_bucket = ((datetime.utcnow().hour * 60) + datetime.utcnow().minute) // 4

if st.toggle("📤 Push raw snapshots") and st.session_state.last_push_min_bucket != push_bucket:
    for sym in ["BTC", "ETH"]:
        df = fetch_live_collector_data(sym, expiry)
        if not df.empty:
            append_raw_snapshot(
                f"{RAW_DIR}/{sym}_{expiry}_snapshots.csv",
                df
            )
    st.session_state.last_push_min_bucket = push_bucket
    st.success("Raw snapshots pushed safely.")
