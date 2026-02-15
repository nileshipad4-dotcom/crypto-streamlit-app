import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import base64
from io import StringIO
from streamlit_autorefresh import st_autorefresh
import time

# =========================================================
# CONFIG
# =========================================================

FIXED_THRESHOLDS = {
    "BTC": 10000,
    "ETH": 15000,
}

DATA_DIR = "data"
RAW_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="OI Time Scanner")
st_autorefresh(interval=60_000, key="refresh")


if "push_enabled" not in st.session_state:
    st.session_state.push_enabled = True



DELTA_API = "https://api.india.delta.exchange/v2/tickers"

GITHUB_TOKEN  = st.secrets["GITHUB_TOKEN"]
CRYPTO_REPO   = "nileshipad4-dotcom/crypto-streamlit-app"
GITHUB_BRANCH = "main"
GITHUB_API    = "https://api.github.com"

CANONICAL_COLS = [
    "call_mark","call_oi","call_volume",
    "call_gamma","call_delta","call_vega",
    "strike_price",
    "put_gamma","put_delta","put_vega",
    "put_volume","put_oi","put_mark",
    "Expiry","timestamp_IST","max_pain"
]

# =========================================================
# SESSION STATE
# =========================================================

if "last_push_bucket" not in st.session_state:
    st.session_state.last_push_bucket = None

if "default_csv_ts" not in st.session_state:
    st.session_state.default_csv_ts = None

# =========================================================
# PRICE
# =========================================================

@st.cache_data(ttl=5)
def get_delta_price(symbol):
    try:
        r = requests.get(DELTA_API, timeout=5).json()["result"]
        return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])
    except Exception:
        return None

# =========================================================
# EXPIRIES
# =========================================================

def get_upcoming_expiry():
    if not os.path.exists(RAW_DIR):
        return None

    today = datetime.utcnow().date()
    expiries = []

    for f in os.listdir(RAW_DIR):
        if f.endswith("_snapshots.csv"):
            try:
                expiry = f.split("_")[1]
                dt = datetime.strptime(expiry, "%d-%m-%Y").date()
                if dt >= today:
                    expiries.append(dt)
            except Exception:
                pass

    if not expiries:
        return None

    return min(expiries).strftime("%d-%m-%Y")



def sync_from_github(repo_path, local_path):
    url = f"{GITHUB_API}/repos/{CRYPTO_REPO}/contents/{repo_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return

    content = base64.b64decode(r.json()["content"]).decode()
    if not content.strip():
        return

    github_df = pd.read_csv(StringIO(content))

    # If local file does not exist → write full file
    if not os.path.exists(local_path):
        github_df.to_csv(local_path, index=False)
        return

    local_df = pd.read_csv(local_path)

    # Append only NEW rows
    key_cols = ["timestamp_IST", "strike_price"]
    merged = github_df.merge(
        local_df[key_cols],
        on=key_cols,
        how="left",
        indicator=True
    )

    new_rows = merged[merged["_merge"] == "left_only"].drop(columns="_merge")

    if new_rows.empty:
        return

    new_rows.to_csv(
        local_path,
        mode="a",
        header=False,
        index=False
    )


def get_target_expiry_plus_one():
    """
    Returns (current expiry + 1 day) in DD-MM-YYYY format.
    Current expiry assumed as nearest future expiry from RAW_DIR.
    """
    expiries = get_all_expiries()
    if not expiries:
        return None

    # nearest expiry (latest in sorted descending list)
    nearest = min(
        expiries,
        key=lambda x: datetime.strptime(x, "%d-%m-%Y")
    )

    dt = datetime.strptime(nearest, "%d-%m-%Y") + timedelta(days=1)
    return dt.strftime("%d-%m-%Y")

def format_for_delta(expiry_ddmmyyyy):
    """
    Converts DD-MM-YYYY → YYYY-MM-DD for Delta API
    """
    dt = datetime.strptime(expiry_ddmmyyyy, "%d-%m-%Y")
    return dt.strftime("%Y-%m-%d")

# -------------------------------------------------
# MAX PAIN (Same as collector.py)
# -------------------------------------------------
def compute_max_pain(df):

    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    n = len(df)
    U = []

    for i in range(n):
        q = -sum(A[i:] * B[i:])
        r = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        s = -sum(M[:i] * L[:i])
        t = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        U.append(round((q + r + s + t) / 10000))

    df["max_pain"] = U
    return df

# =========================================================
# LOAD CLEAN OI HISTORY
# =========================================================

def load_data(symbol, expiry):
    path = f"{RAW_DIR}/{symbol}_{expiry}_snapshots.csv"

    # ---- HARD GUARD ----
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=CANONICAL_COLS)

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=CANONICAL_COLS)

    if df.empty or "timestamp_IST" not in df.columns:
        return pd.DataFrame(columns=CANONICAL_COLS)

    df["_row"] = range(len(df))

    raw = pd.to_datetime(df["timestamp_IST"], format="%H:%M", errors="coerce")
    raw = raw.dropna()

    if raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLS)

    base = datetime(2000, 1, 1)
    out, last, day = [], None, 0

    for t in raw:
        if last is not None and t < last:
            day += 1
        out.append(base + timedelta(days=day, hours=t.hour, minutes=t.minute))
        last = t

    df = df.loc[raw.index].copy()
    df["timestamp_IST"] = out

    # remove duplicate strike rows per timestamp
    df = df.drop_duplicates(subset=["timestamp_IST", "strike_price"])

    return df


def get_all_expiries():
    expiries = set()

    if not os.path.exists(RAW_DIR):
        return []

    for f in os.listdir(RAW_DIR):
        if f.endswith("_snapshots.csv"):
            try:
                expiry = f.split("_")[1]
                datetime.strptime(expiry, "%d-%m-%Y")
                expiries.add(expiry)
            except Exception:
                pass

    return sorted(
        expiries,
        key=lambda x: datetime.strptime(x, "%d-%m-%Y"),
        reverse=True
    )


def get_latest_csv_time(symbol, expiry):
    """
    Returns the true latest timestamp from CSV,
    correctly handling midnight rollover.
    """
    path = f"{RAW_DIR}/{symbol}_{expiry}_snapshots.csv"
    if not os.path.exists(path):
        return "—"

    try:
        df = pd.read_csv(path, usecols=["timestamp_IST"])
        if df.empty:
            return "—"

        raw = pd.to_datetime(df["timestamp_IST"], format="%H:%M", errors="coerce")
        raw = raw.dropna()
        if raw.empty:
            return "—"

        base = datetime(2000, 1, 1)
        out, last, day = [], None, 0

        for t in raw:
            if last is not None and t < last:
                day += 1  # midnight rollover
            out.append(base + timedelta(days=day, hours=t.hour, minutes=t.minute))
            last = t

        latest = max(out)
        return latest.strftime("%H:%M")

    except Exception:
        return "—"

def get_csv_times(df):
    """
    Returns CSV timestamps sorted by actual datetime (descending),
    handles midnight rollover correctly.
    """
    times = (
        df[["timestamp_IST"]]
        .drop_duplicates()
        .sort_values("timestamp_IST", ascending=False)
        ["timestamp_IST"]
        .tolist()
    )
    return times


def fetch_live_option_chain_totals(symbol, expiry):
    """
    Fetch live option chain with TOTAL OI and TOTAL VOLUME
    (no deltas, no calculations)
    """
    url = (
        f"{DELTA_API}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={symbol}"
        f"&expiry_date={format_for_delta(expiry)}"
    )

    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json().get("result", [])
    if not data:
        return pd.DataFrame()

    df = pd.json_normalize(data)

    df = df[[
        "strike_price",
        "contract_type",
        "oi_contracts",
        "volume"
    ]]

    calls = (
        df[df["contract_type"] == "call_options"]
        .rename(columns={
            "oi_contracts": "Call OI",
            "volume": "Call Volume"
        })
        .drop(columns=["contract_type"])
    )

    puts = (
        df[df["contract_type"] == "put_options"]
        .rename(columns={
            "oi_contracts": "Put OI",
            "volume": "Put Volume"
        })
        .drop(columns=["contract_type"])
    )

    out = pd.merge(
        calls,
        puts,
        on="strike_price",
        how="outer"
    )

    out = out.rename(columns={"strike_price": "Strike"})

    return (
        out
        .fillna(0)
        .astype({
            "Strike": int,
            "Call OI": int,
            "Call Volume": int,
            "Put OI": int,
            "Put Volume": int
        })
        .sort_values("Strike")
        .reset_index(drop=True)
    )


def build_delta_table(df_hist, ts1, ts2, use_live, df_live=None):
    """
    Returns strike-wise delta OI & Volume table.
    """
    d1 = df_hist[df_hist["timestamp_IST"] == ts1]
    if d1.empty:
        return pd.DataFrame()

    if use_live:
        if df_live is None or df_live.empty:
            return pd.DataFrame()
        d2 = df_live.copy()
    else:
        d2 = df_hist[df_hist["timestamp_IST"] == ts2]
        if d2.empty:
            return pd.DataFrame()

    # align
    d1 = d1.copy()
    d2 = d2.copy()
    d1["strike_price"] = d1["strike_price"].astype(int)
    d2["strike_price"] = d2["strike_price"].astype(int)

    m = pd.merge(
        d1, d2, on="strike_price", suffixes=("_1", "_2")
    )

    if m.empty:
        return pd.DataFrame()

    for c in ["call_oi_1","call_oi_2","put_oi_1","put_oi_2",
              "call_volume_1","call_volume_2","put_volume_1","put_volume_2"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    out = pd.DataFrame({
        "Strike": m["strike_price"].astype(int),
        "Δ Call OI": (m["call_oi_2"] - m["call_oi_1"]).astype(int),
        "Δ Put OI":  (m["put_oi_2"]  - m["put_oi_1"]).astype(int),
        "Δ Call Vol": (m["call_volume_2"] - m["call_volume_1"]).astype(int),
        "Δ Put Vol":  (m["put_volume_2"]  - m["put_volume_1"]).astype(int),
    })


    return out.sort_values("Strike").reset_index(drop=True)

def style_delta_table(df, price):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    # --- highlight rows surrounding current price (BLUE) ---
    if price:
        below = df[df["Strike"] <= price]
        above = df[df["Strike"] >= price]

        highlight_rows = set()

        if not below.empty:
            highlight_rows.add(below.index[-1])

        if not above.empty:
            highlight_rows.add(above.index[0])

        for i in highlight_rows:
            styles.loc[i, :] += (
                "background-color:#1e90ff;"
                "color:white;"
                "font-weight:bold;"
            )

    # --- top 2 absolute values per column (ORANGE TEXT ONLY) ---
    for col in df.columns:
        if col == "Strike":
            continue

        ranks = df[col].abs().rank(method="first", ascending=False)

        for i in df.index:
            if ranks.loc[i] in (1, 2):
                styles.loc[i, col] += "color:orange;font-weight:bold;"

    return df.style.apply(lambda _: styles, axis=None)

# =========================================================
# WINDOW ENGINE
# =========================================================
def build_live_row_from_last_snapshot(df_hist, df_live, last_time):
    """
    Compare last CSV snapshot vs LIVE snapshot and
    return one synthetic row (dict) for main table.
    """
    if df_hist.empty or df_live.empty:
        return None

    # last snapshot at last_time
    d1 = df_hist[df_hist["timestamp_IST"] <= last_time].sort_values("timestamp_IST").tail(1)
    if d1.empty:
        return None

    # live snapshot (same structure)
    # live snapshot (same structure)
    d2 = df_live.copy()
    
    # 🔑 FIX: align merge key dtype
    d1 = d1.copy()
    d1["strike_price"] = d1["strike_price"].astype(int)
    d2["strike_price"] = d2["strike_price"].astype(int)


    m = pd.merge(
        d1,
        d2,
        on="strike_price",
        suffixes=("_1", "_2")
    )

    if m.empty:
        return None

    # ignore extreme strikes (same logic as build_row)
    strikes = sorted(m["strike_price"].unique())
    if len(strikes) <= 4:
        return None
    m = m[m["strike_price"].isin(strikes[2:-2])]
    # 🔑 FIX: force numeric OI columns
    for c in ["call_oi_1", "call_oi_2", "put_oi_1", "put_oi_2"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    # delta from last snapshot → live
    m["CE"] = m["call_oi_2"] - m["call_oi_1"]
    m["PE"] = m["put_oi_2"] - m["put_oi_1"]

    agg = (
        m.groupby("strike_price", as_index=False)
         .agg({"CE": "sum", "PE": "sum"})
    )

    ce = agg.sort_values("CE", ascending=False)
    pe = agg.sort_values("PE", ascending=False)

    # 🔑 LIVE rows may not have full depth — allow partial
    if ce.empty or pe.empty:
        return None
    
    # fallback if only 1 strike exists
    def safe_row(df, idx):
        if len(df) > idx:
            return df.iloc[idx]
        return df.iloc[0]
    
    ce1 = safe_row(ce, 0)
    ce2 = safe_row(ce, 1)
    pe1 = safe_row(pe, 0)
    pe2 = safe_row(pe, 1)


    sum_ce = int(agg["CE"].sum() / 100)
    sum_pe = int(agg["PE"].sum() / 100)

    return {
        "TIME": f"{last_time:%H:%M} → LIVE",
        "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
        "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
        "Σ ΔCE OI": sum_ce,
        "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
        "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",
        "Σ ΔPE OI": sum_pe,
        "Δ (PE − CE)": sum_pe - sum_ce,
        "_ce1": ce.iloc[0].CE,
        "_ce2": ce.iloc[1].CE,
        "_pe1": pe.iloc[0].PE,
        "_pe2": pe.iloc[1].PE,
    }


def build_all_windows(df, gap):
    if df.empty or "timestamp_IST" not in df.columns:
        return []

    times = (
        df.sort_values("timestamp_IST")
          .drop_duplicates("timestamp_IST")["timestamp_IST"]
          .tolist()
    )

    windows, i = [], 0
    while i < len(times) - 1:
        t1 = times[i]
        t2 = next(
            (t for t in times[i + 1:] if t >= t1 + timedelta(minutes=gap)),
            None
        )
        if not t2:
            break
        windows.append((t1, t2))
        i = times.index(t2)

    return windows


    windows, i = [], 0
    while i < len(times)-1:
        t1 = times[i]
        t2 = next((t for t in times[i+1:] if t >= t1 + timedelta(minutes=gap)), None)
        if not t2:
            break
        windows.append((t1,t2))
        i = times.index(t2)
    return windows


def build_row(df, t1, t2, live=False):
    d1 = df[df["timestamp_IST"]==t1]
    d2 = df[df["timestamp_IST"]==t2]
    if d1.empty or d2.empty:
        return None

    m = pd.merge(d1,d2,on="strike_price",suffixes=("_1","_2"))
    strikes = sorted(m["strike_price"].unique())
    if len(strikes)<=4:
        return None
    m = m[m["strike_price"].isin(strikes[2:-2])]

    m["CE"] = m["call_oi_2"] - m["call_oi_1"]
    m["PE"] = m["put_oi_2"] - m["put_oi_1"]

    # ---- AGGREGATE BY STRIKE TO AVOID DUPLICATES ----
    agg = (
        m.groupby("strike_price", as_index=False)
         .agg({"CE": "sum", "PE": "sum"})
    )
    
    ce = agg.sort_values("CE", ascending=False)
    pe = agg.sort_values("PE", ascending=False)

    if len(ce)<2 or len(pe)<2:
        return None

    sum_ce = int(agg["CE"].sum() / 100)
    sum_pe = int(agg["PE"].sum() / 100)


    return {
        "TIME": f"{t1:%H:%M} - {t2:%H:%M}" + (" ⏳" if live else ""),
        "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
        "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
        "Σ ΔCE OI": sum_ce,
        "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
        "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",
        "Σ ΔPE OI": sum_pe,
        "Δ (PE − CE)": sum_pe - sum_ce,
        "_ce1": ce.iloc[0].CE,
        "_ce2": ce.iloc[1].CE,
        "_pe1": pe.iloc[0].PE,
        "_pe2": pe.iloc[1].PE,
        "_end": t2
    }


def process_windows(df, gap):
    rows = []
    windows = build_all_windows(df, gap)

    for t1, t2 in windows:
        r = build_row(df, t1, t2)
        if r:
            rows.append(r)

    # ❌ NO LIVE / RUNNING WINDOW
    # ❌ NO partial delta calculation

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("_end").drop(columns="_end")
    return out



def build_csv_vs_live_row(df_hist, df_live, ts):
    d1 = df_hist[df_hist["timestamp_IST"] == ts]
    if d1.empty or df_live.empty:
        return None

    d1 = d1.copy()
    d2 = df_live.copy()

    d1["strike_price"] = d1["strike_price"].astype(int)
    d2["strike_price"] = d2["strike_price"].astype(int)

    m = pd.merge(d1, d2, on="strike_price", suffixes=("_1", "_2"))
    if m.empty:
        return None

    # 🔑 FIX: force numeric OI columns
    for c in ["call_oi_1", "call_oi_2", "put_oi_1", "put_oi_2"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    strikes = sorted(m["strike_price"].unique())
    if len(strikes) <= 4:
        return None
    m = m[m["strike_price"].isin(strikes[2:-2])]

    m["CE"] = m["call_oi_2"] - m["call_oi_1"]
    m["PE"] = m["put_oi_2"] - m["put_oi_1"]

    agg = (
        m.groupby("strike_price", as_index=False)
         .agg({"CE": "sum", "PE": "sum"})
    )

    ce = agg.sort_values("CE", ascending=False)
    pe = agg.sort_values("PE", ascending=False)

    if len(ce) < 2 or len(pe) < 2:
        return None

    sum_ce = int(agg["CE"].sum() / 100)
    sum_pe = int(agg["PE"].sum() / 100)

    return {
        "TIME": f"{ts:%H:%M} → LIVE",
        "MAX CE 1": f"{int(ce.iloc[0].strike_price)}:- {int(ce.iloc[0].CE)}",
        "MAX CE 2": f"{int(ce.iloc[1].strike_price)}:- {int(ce.iloc[1].CE)}",
        "Σ ΔCE OI": sum_ce,
        "MAX PE 1": f"{int(pe.iloc[0].strike_price)}:- {int(pe.iloc[0].PE)}",
        "MAX PE 2": f"{int(pe.iloc[1].strike_price)}:- {int(pe.iloc[1].PE)}",
        "Σ ΔPE OI": sum_pe,
        "Δ (PE − CE)": sum_pe - sum_ce,
    }

    
# =========================================================
# SIDE TABLE: LARGE OI EXTRACTOR
# =========================================================

def extract_big_oi(df, threshold=10000):
    ce_rows = []
    pe_rows = []

    def parse(cell):
        # "84000:- 12002" → (84000, 12002)
        try:
            strike, val = cell.split(":-")
            return int(strike.strip()), int(val.strip())
        except Exception:
            return None, None

    for _, row in df.iterrows():
        for col in ["MAX CE 1", "MAX CE 2"]:
            s, v = parse(row[col])
            if s is not None and abs(v) > threshold:
                ce_rows.append((s, v))

        for col in ["MAX PE 1", "MAX PE 2"]:
            s, v = parse(row[col])
            if s is not None and abs(v) > threshold:
                pe_rows.append((s, v))

    ce_df = pd.DataFrame(ce_rows, columns=["Strike", "ΔOI"])
    pe_df = pd.DataFrame(pe_rows, columns=["Strike", "ΔOI"])

    ce_df = ce_df.sort_values("Strike").reset_index(drop=True)
    pe_df = pe_df.sort_values("Strike").reset_index(drop=True)

    return ce_df, pe_df

def mark_price_neighbors(df, price):
    """
    Returns set of row indices to highlight based on price position.
    """
    if df.empty or "Strike" not in df:
        return set()

    strikes = df["Strike"].dropna().values
    if len(strikes) == 0:
        return set()

    below = df[df["Strike"] <= price]
    above = df[df["Strike"] >= price]

    highlight = set()

    if not below.empty:
        highlight.add(below.index[-1])

    if not above.empty:
        highlight.add(above.index[0])

    return highlight


# =========================================================
# HIGHLIGHTING
# =========================================================


def highlight_table(df, price):
    cols = [
        "TIME","MAX CE 1","MAX CE 2","Σ ΔCE OI",
        "MAX PE 1","MAX PE 2","Σ ΔPE OI","Δ (PE − CE)"
    ]

    # 🔑 SAFETY GUARD (THIS FIXES YOUR ERROR)
    if df.empty or not all(c in df.columns for c in cols):
        return df

    styles = pd.DataFrame("", index=df.index, columns=cols)

    # ---------- EXISTING HIGHLIGHTS ----------
    highlight_map = [
        ("MAX CE 1","_ce1"),
        ("MAX CE 2","_ce2"),
        ("MAX PE 1","_pe1"),
        ("MAX PE 2","_pe2"),
    ]

    for col, num in highlight_map:
        if num not in df.columns:
            continue

        v = df[num].abs()
        ranks = v.rank(method="first", ascending=False)

        for i in df.index:
            if ranks.loc[i] == 1:
                styles.loc[i, col] += (
                    "background-color:#ffa500;color:white;font-weight:bold;"
                )
            elif ranks.loc[i] == 2:
                styles.loc[i, col] += (
                    "background-color:#ff4d4d;font-weight:bold;"
                )

    for i in df.index:
        diff = df.loc[i, "Δ (PE − CE)"]
        if diff > 0:
            color = "green"
        elif diff < 0:
            color = "red"
        else:
            continue

        styles.loc[i, "Σ ΔCE OI"] += f"color:{color};font-weight:bold;"
        styles.loc[i, "Σ ΔPE OI"] += f"color:{color};font-weight:bold;"
        styles.loc[i, "Δ (PE − CE)"] += f"color:{color};font-weight:bold;"

    # ---------- 2.5% PRICE PROXIMITY ----------
    # ---------- 2.5% PRICE + VALUE FILTER ----------
    if price and price > 0:
        low, high = price * 0.994, price * 1.006

        def extract(cell):
            # "84000:- 12002" → (84000, 12002)
            try:
                s, v = cell.split(":-")
                return int(s.strip()), int(v.strip())
            except Exception:
                return None, None

        for i in df.index:
            for col in ["MAX CE 1","MAX CE 2","MAX PE 1","MAX PE 2"]:
                strike, val = extract(df.loc[i, col])

                if (
                    strike is not None
                    and val is not None
                    and abs(val) > 2000
                    and low <= strike <= high
                ):
                    styles.loc[i, col] += (
                        "background-color:#1e90ff;"
                        "color:white;"
                        "font-weight:bold;"
                    )

    return df[cols].style.apply(lambda _: styles, axis=None)

# =========================================================
# RAW COLLECTOR
# =========================================================

def get_ist():
    return (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")

def fetch_live(symbol, expiry):
    url = (
        f"{DELTA_API}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={symbol}"
        f"&expiry_date={format_for_delta(expiry)}"
    )
    r = requests.get(url,timeout=15)
    if r.status_code!=200:
        return pd.DataFrame()

    df = pd.json_normalize(r.json().get("result",[]))
    if df.empty:
        return df

    df = df[[
        "strike_price","contract_type",
        "mark_price","oi_contracts","volume",
        "greeks.gamma","greeks.delta","greeks.vega"
    ]]

    df.rename(columns={
        "greeks.gamma":"gamma",
        "greeks.delta":"delta",
        "greeks.vega":"vega"
    }, inplace=True)

    calls = df[df.contract_type=="call_options"].copy()
    puts  = df[df.contract_type=="put_options"].copy()

    calls = calls.rename(columns={
        "mark_price":"call_mark","oi_contracts":"call_oi",
        "volume":"call_volume","gamma":"call_gamma",
        "delta":"call_delta","vega":"call_vega"
    }).drop(columns=["contract_type"])

    puts = puts.rename(columns={
        "mark_price":"put_mark","oi_contracts":"put_oi",
        "volume":"put_volume","gamma":"put_gamma",
        "delta":"put_delta","vega":"put_vega"
    }).drop(columns=["contract_type"])

    m = pd.merge(calls,puts,on="strike_price",how="outer")
    m["Expiry"]=expiry
    m["timestamp_IST"]=get_ist()
    m = compute_max_pain(m)


    return m.reindex(columns=CANONICAL_COLS)

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def get_bucket_and_remaining():
    """
    4-minute bucket aligned to IST clock.
    Push is allowed once per bucket.
    """
    now = get_ist_now()

    total_seconds = (now.minute * 60) + now.second
    bucket_len = 4 * 60

    bucket_id = total_seconds // bucket_len
    seconds_remaining = bucket_len - (total_seconds % bucket_len)

    return bucket_id, seconds_remaining


def append_raw(path, df, retries=3):
    url = f"{GITHUB_API}/repos/{CRYPTO_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    for attempt in range(retries):
        r = requests.get(url, headers=headers)
        sha, old = None, ""

        if r.status_code == 200:
            sha = r.json()["sha"]
            old = base64.b64decode(r.json()["content"]).decode()

        new = df.to_csv(index=False, header=(old == ""))
        payload = {
            "message": "raw snapshot",
            "content": base64.b64encode((old + new).encode()).decode(),
            "branch": GITHUB_BRANCH
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, headers=headers, json=payload)

        if resp.status_code in (200, 201):
            return  # ✅ success

        if resp.status_code != 409:
            st.error(f"❌ GitHub push failed: {resp.status_code}")
            st.code(resp.text)
            st.stop()


        # 🔁 SHA conflict → retry
        time.sleep(1)

    raise RuntimeError("GitHub push failed after retries (SHA conflict)")


# =========================================================
# UI
# =========================================================

st.title("BTC & ETH OI Change Scanner")

btc_p = get_delta_price("BTC")
eth_p = get_delta_price("ETH")

# =========================================================
# SIDEBAR — ALWAYS VISIBLE PRICES
# =========================================================
with st.sidebar:
    st.markdown("### 📈 Live Prices")

    st.metric(
        "BTC",
        f"{btc_p:,.2f}" if btc_p else "—",
        delta=None
    )

    st.metric(
        "ETH",
        f"{eth_p:,.2f}" if eth_p else "—",
        delta=None
    )

    st.markdown("---")

c1,c2 = st.columns(2)
c1.metric("BTC Price", f"{btc_p:,.2f}" if btc_p else "—")
c2.metric("ETH Price", f"{eth_p:,.2f}" if eth_p else "—")

c_exp, c_gap, c_thr = st.columns([2,1,1])

with c_exp:
    expiry = get_target_expiry_plus_one()

    if not expiry:
        st.error("No base expiry found in data/raw")
        st.stop()

    st.caption(f"📅 Using expiry (Current +1): **{expiry}**")



# ---------- SYNC LATEST CSV FROM GITHUB ----------
for sym in ["BTC", "ETH"]:
    path = f"{RAW_DIR}/{sym}_{expiry}_snapshots.csv"
    sync_from_github(path, path)

with c_gap:
    gap = st.selectbox("Min Gap (minutes)", [5,10,15,20,30,45,60], index=2)





for sym in ["BTC", "ETH"]:
    st.subheader(sym)
    # Per-symbol default threshold
    
    df_hist = load_data(sym, expiry)
    df = process_windows(df_hist, gap)
    
    # ---------- ADD LIVE ROW ----------
    if not df.empty:
        # 🔑 extract END time of last window shown

        last_time = df_hist["timestamp_IST"].max()

    
        df_live = fetch_live(sym, expiry)
    
        live_row = build_live_row_from_last_snapshot(
            df_hist,
            df_live,
            last_time
        )

    
        if live_row:
            df = pd.concat(
                [df, pd.DataFrame([live_row])],
                ignore_index=True
            )


    main_col, side_col = st.columns([3, 1])

    # ---------------- MAIN TABLE ----------------
    with main_col:
        price = btc_p if sym == "BTC" else eth_p
        
        st.dataframe(
            highlight_table(df, price),
            use_container_width=True
        )


    # ---------------- SIDE TABLE ----------------
    with side_col:
        if not df.empty:
            threshold = FIXED_THRESHOLDS[sym]
            ce_df, pe_df = extract_big_oi(df, threshold=threshold)


            max_len = max(len(ce_df), len(pe_df))
            ce_df = ce_df.reindex(range(max_len))
            pe_df = pe_df.reindex(range(max_len))

            side_table = pd.DataFrame({
                "CE > 10k": ce_df.apply(
                    lambda r: f"{int(r.Strike)} : {int(r['ΔOI'])}"
                    if pd.notna(r.Strike) else "",
                    axis=1
                ),
                "PE > 10k": pe_df.apply(
                    lambda r: f"{int(r.Strike)} : {int(r['ΔOI'])}"
                    if pd.notna(r.Strike) else "",
                    axis=1
                ),
            })

            # Determine current price
            price = btc_p if sym == "BTC" else eth_p
            
            ce_hi = mark_price_neighbors(ce_df, price)
            pe_hi = mark_price_neighbors(pe_df, price)
            
            def style_side(row):
                styles = ["", ""]
            
                if row.name in ce_hi:
                    styles[0] = "font-weight:bold;color:orange"
            
                if row.name in pe_hi:
                    styles[1] = "font-weight:bold;color:orange"
            
                return styles


            st.markdown("**Large OI Changes**")
            st.dataframe(
                side_table.style.apply(style_side, axis=1),
                use_container_width=True,
                height=320
            )




    if not df.empty:
        times = df["TIME"].tolist()
    
        # 🔑 CAPTURE DEFAULT CSV TIME FROM LAST MAIN ROW
        last_time_label = times[-1]
        try:
            if "-" in last_time_label:
                st.session_state.default_csv_ts = last_time_label.split("-")[-1].strip()
            elif "→" in last_time_label:
                st.session_state.default_csv_ts = last_time_label.split("→")[0].strip()
        except Exception:
            pass

        f,t = st.columns(2)
        with f:
            t1=st.selectbox(f"From ({sym})",times,index=0,key=f"{sym}f")
        with t:
            t2=st.selectbox(f"To ({sym})",times,index=len(times)-1,key=f"{sym}t")

        i1,i2=times.index(t1),times.index(t2)
        s=df.iloc[min(i1,i2):max(i1,i2)+1]

        ce,pe,d = int(s["Σ ΔCE OI"].sum()), int(s["Σ ΔPE OI"].sum()), int(s["Δ (PE − CE)"].sum())
        def c(v): return "red" if v>0 else "green" if v<0 else "black"

        a,b,c3 = st.columns(3)
        a.markdown(f"**△ CE:** <span style='color:{c(ce)}'>{ce}</span>",unsafe_allow_html=True)
        b.markdown(f"**△ PE:** <span style='color:{c(pe)}'>{pe}</span>",unsafe_allow_html=True)
        c3.markdown(f"**△ (PE − CE):** <span style='color:{c(d)}'>{d}</span>",unsafe_allow_html=True)






st.markdown("---")
st.header("📊 CSV vs LIVE (Delta API)")

df_hist_btc = load_data("BTC", expiry)

csv_times = (
    df_hist_btc["timestamp_IST"]
    .drop_duplicates()
    .sort_values(ascending=False)
    .tolist()
)

# 🔑 DEFAULT INDEX FROM MAIN TABLE WINDOW
default_idx = 0
if st.session_state.default_csv_ts:
    for i, t in enumerate(csv_times):
        if t.strftime("%H:%M") == st.session_state.default_csv_ts:
            default_idx = i
            break

selected_ts = st.selectbox(
    "Select CSV Time",
    csv_times,
    index=default_idx,
    format_func=lambda x: x.strftime("%H:%M")
)


rows = []

for sym in ["BTC", "ETH"]:
    df_hist = load_data(sym, expiry)

    # LIVE DATA FROM DELTA API
    df_live = fetch_live(sym, expiry)

    r = build_csv_vs_live_row(df_hist, df_live, selected_ts)
    if r:
        r["SYMBOL"] = sym
        rows.append(r)

if rows:
    out = pd.DataFrame(rows).set_index("SYMBOL")

    st.dataframe(
        out,
        use_container_width=True,
        height=140
    )
else:
    st.info("No LIVE data available from Delta API")


# =========================================================
# RAW PUSH
# =========================================================



bucket = ((datetime.utcnow().hour * 60) + datetime.utcnow().minute) // 4
col_t, col_c = st.columns([1, 2])

with col_t:
    st.toggle("📤 Push raw snapshots", key="push_enabled")

bucket, remaining = get_bucket_and_remaining()
mm, ss = divmod(remaining, 60)

# ---------- SYNC LATEST CSV FROM GITHUB ----------



latest_btc = get_latest_csv_time("BTC", expiry)
latest_eth = get_latest_csv_time("ETH", expiry)

with col_c:
    if st.session_state.push_enabled:
        st.markdown(
            f"**⏱ Next data in:** `{mm:02d}:{ss:02d}`  |  "
            f"**Last CSV:** BTC `{latest_btc}` · ETH `{latest_eth}`"
        )
    else:
        st.markdown(
            f"⏸ Snapshot push paused  |  "
            f"**Last CSV:** BTC `{latest_btc}` · ETH `{latest_eth}`"
        )



st.markdown("---")
st.header("⏱ Strike-wise Delta (CSV / LIVE)")

for sym in ["BTC", "ETH"]:
    st.subheader(sym)

    df_hist = load_data(sym, expiry)
    if df_hist.empty:
        st.info("No CSV data")
        continue

    csv_times = get_csv_times(df_hist)
    if len(csv_times) < 1:
        st.info("Not enough CSV timestamps")
        continue

    # --- timestamp selectors ---
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        ts1 = st.selectbox(
            f"{sym} Timestamp 1",
            csv_times,
            index=0,
            format_func=lambda x: x.strftime("%H:%M"),
            key=f"{sym}_ts1"
        )

    with col2:
        ts2 = st.selectbox(
            f"{sym} Timestamp 2",
            csv_times,
            index=min(1, len(csv_times)-1),
            format_func=lambda x: x.strftime("%H:%M"),
            key=f"{sym}_ts2"
        )

    with col3:
        use_live = st.toggle(
            "Use LIVE vs TS1",
            value=True,
            key=f"{sym}_live_toggle"
        )

    df_live = fetch_live(sym, expiry) if use_live else None

    delta_df = build_delta_table(
        df_hist, ts1, ts2, use_live, df_live
    )

    if delta_df.empty:
        st.info("No delta data available")
        continue

    price = btc_p if sym == "BTC" else eth_p

    st.dataframe(
        style_delta_table(delta_df, price),
        use_container_width=True,
        height=420
    )








if (
    st.session_state.push_enabled
    and 120 < remaining < 180
    and st.session_state.last_push_bucket != bucket
):
    for sym in ["BTC", "ETH"]:
        df_live = fetch_live(sym, expiry)
    
        if not df_live.empty:
            append_raw(
                f"{RAW_DIR}/{sym}_{expiry}_snapshots.csv",
                df_live
            )

    st.session_state.last_push_bucket = bucket
    st.success("Raw snapshots appended successfully (GitHub confirmed).")

