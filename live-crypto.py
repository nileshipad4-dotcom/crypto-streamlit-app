# crypto compare — FINAL COMPLETE VERSION (EXPIRY + MP + PCR + LIVE OI/VOLUME DELTAS)

import streamlit as st
import pandas as pd
import requests
import calendar
import os
from datetime import datetime, timedelta, date
from streamlit_autorefresh import st_autorefresh
import base64
from io import StringIO





# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Strike-wise Comparison + Live Snapshot")

# -------------------------------------------------
# AUTO REFRESH (60s)
# -------------------------------------------------
if st_autorefresh(interval=60_000, key="auto_refresh"):
    st.cache_data.clear()

if "last_push_ts" not in st.session_state:
    st.session_state.last_push_ts = None

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
ASSETS = ["BTC", "ETH"]
BASE_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "nileshipad4-dotcom/crypto-streamlit-app/"
    "main/data/"
)

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
CRYPTO_REPO = "nileshipad4-dotcom/crypto-streamlit-app"
GITHUB_BRANCH = "main"
GITHUB_API = "https://api.github.com"

CANONICAL_COLS = [
    "call_mark","call_oi","call_volume",
    "call_gamma","call_delta","call_vega",
    "strike_price",
    "put_gamma","put_delta","put_vega",
    "put_volume","put_oi","put_mark",
    "Expiry","timestamp_IST","max_pain"
]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_ist_hhmm():
    return get_ist_datetime().strftime("%H:%M")

def safe_ratio(a, b):
    try:
        if b == 0 or pd.isna(b):
            return None
        return float(a) / float(b)
    except:
        return None

def extract_timestamps_from_local_csv(underlying, expiry):
    url = f"{BASE_RAW_URL}{underlying}_{expiry}.csv"

    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"CSV load failed: {e}")
        return []

    if "timestamp_IST" not in df.columns:
        st.error("timestamp_IST column missing in CSV")
        return []

    df["timestamp_IST"] = pd.to_datetime(df["timestamp_IST"], errors="coerce")

    times = (
        df["timestamp_IST"]
        .dropna()
        .dt.strftime("%H:%M")
        .unique()
        .tolist()
    )

    if len(times) < 2:
        st.error("CSV loaded but timestamps < 2")
        return []

    pivot = 17 * 60 + 31

    def sort_key(t):
        h, m = map(int, t.split(":"))
        return (pivot - (h * 60 + m)) % 1440

    return sorted(times, key=sort_key)


@st.cache_data(ttl=15)
def fetch_live_collector_data(underlying, expiry):
    url = (
        f"{API_BASE}"
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

    merged["timestamp_IST"] = get_ist_hhmm()
    merged["Expiry"] = expiry

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

# -------------------------------------------------
# EXPIRY LOGIC
# -------------------------------------------------
def get_expiries():
    ist_now = get_ist_datetime()
    today = ist_now.date()

    if ist_now.time() <= datetime.strptime("17:30", "%H:%M").time():
        latest_valid = today
    else:
        latest_valid = today + timedelta(days=1)

    expiries = {latest_valid}

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
        lf = max(w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY] != 0)
        expiries.add(date(y, m, lf))

    return sorted([d.strftime("%d-%m-%Y") for d in expiries])

# -------------------------------------------------
# CONTROLS
# -------------------------------------------------
c1, c2, c3 = st.columns([1, 6, 3])

with c1:
    asset = st.selectbox("", ASSETS)

with c3:
    expiry_list = get_expiries()
    selected_expiry = st.selectbox("Expiry", expiry_list)
    csv_mode = st.toggle("📁 CSV Mode (T1 vs T2)", value=False)

with c2:
    if st.button("⏱ Load Timestamps"):
        st.session_state.timestamps = extract_timestamps_from_local_csv(asset, selected_expiry)

if "timestamps" not in st.session_state or not st.session_state.timestamps:
    st.session_state.timestamps = extract_timestamps_from_local_csv(asset, selected_expiry)

timestamps = st.session_state.timestamps

if len(timestamps) < 2:
    st.warning("Not enough timestamps in CSV.")
    st.stop()

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
if "ref_price" not in st.session_state:
    st.session_state.ref_price = float(prices[asset])


def append_csv_to_github(path, new_df, commit_msg):
    """
    Appends new rows to an existing CSV on GitHub.
    Creates file if it does not exist.
    """
    url = f"{GITHUB_API}/repos/{CRYPTO_REPO}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # 1️⃣ Check if file exists
    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        # File exists → append
        content = base64.b64decode(r.json()["content"]).decode()
        sha = r.json()["sha"]

        old_df = pd.read_csv(StringIO(content))
        
        # 🔒 FORCE CANONICAL ORDER ON OLD DATA
        old_df = old_df.reindex(columns=CANONICAL_COLS)
        
        # 🔒 FORCE CANONICAL ORDER ON NEW DATA
        new_df = new_df.reindex(columns=CANONICAL_COLS)
        
        # ✅ SAFE APPEND
        final_df = pd.concat([old_df, new_df], ignore_index=True)


    else:
        # File does not exist → create new
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

now_ts = get_ist_hhmm()

if st.session_state.last_push_ts != now_ts:

    expiry_to_use = get_expiries()[0]  # nearest valid expiry

    for underlying in ["BTC", "ETH"]:

        df_live = fetch_live_collector_data(underlying, expiry_to_use)

        if df_live.empty:
            continue

        df_live = compute_max_pain_collector(df_live)

        df_live = df_live.reindex(columns=CANONICAL_COLS)

        github_path = f"data/{underlying}_{expiry_to_use}.csv"
        commit_msg = f"{underlying} snapshot @ {now_ts} IST"

        append_csv_to_github(
            path=github_path,
            new_df=df_live,
            commit_msg=commit_msg
        )

    st.session_state.last_push_ts = now_ts

# ==========================================
# OI Weighted Strike Range Settings
# ==========================================
st.subheader("📌 OI Weighted Strike Range Settings")

r1, r2 = st.columns([2, 1])

with r1:
    ref_price = st.number_input(
        "Reference Price (Default = LTP)",
        value=st.session_state.ref_price,
        step=100.0
    )

    st.session_state.ref_price = ref_price


with r2:
    pct_range = st.number_input(
        "Range %",
        value=3.0,
        step=0.5
    )


# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
pcr_rows = []

summary_rows = []

# Placeholder for summary table (will render after data is collected)
summary_placeholder = st.empty()

for UNDERLYING in ASSETS:

    base_price = ref_price if asset == UNDERLYING else prices[UNDERLYING]
    
    low = base_price * (1 - pct_range / 100)
    high = base_price * (1 + pct_range / 100)

    csv_url = f"{BASE_RAW_URL}{UNDERLYING}_{selected_expiry}.csv"

    st.write("📄 Loading CSV:", csv_url)
    try:
        df_raw = pd.read_csv(csv_url)
    except Exception as e:
        st.warning(f"Failed to load CSV for {UNDERLYING} {selected_expiry}: {e}")
        continue

    df = pd.DataFrame({
        "strike_price": pd.to_numeric(df_raw["strike_price"], errors="coerce"),
        "value": pd.to_numeric(df_raw["max_pain"], errors="coerce"),
        "call_oi": pd.to_numeric(df_raw["call_oi"], errors="coerce"),
        "put_oi": pd.to_numeric(df_raw["put_oi"], errors="coerce"),
        "call_vol": pd.to_numeric(df_raw["call_volume"], errors="coerce"),
        "put_vol": pd.to_numeric(df_raw["put_volume"], errors="coerce"),
        "timestamp": df_raw["timestamp_IST"].astype(str).str[:5],
    }).dropna()

    # ---------- MP SNAPSHOTS ----------
    df_t1 = df[df["timestamp"] == t1].groupby("strike_price")["value"].sum()
    df_t2 = df[df["timestamp"] == t2].groupby("strike_price")["value"].sum()

    merged = pd.concat([df_t1, df_t2], axis=1)
    merged.columns = [f"MP ({t1})", f"MP ({t2})"]
    merged["△ MP 2"] = merged.iloc[:, 0] - merged.iloc[:, 1]
    merged = merged.reset_index()

    # ---------- LIVE API ----------
    df_live = pd.json_normalize(
        requests.get(
            f"{API_BASE}?contract_types=call_options,put_options"
            f"&underlying_asset_symbols={UNDERLYING}"
            f"&expiry_date={selected_expiry}",
            timeout=20,
        ).json()["result"]
    )

    df_live["oi_contracts"] = pd.to_numeric(df_live["oi_contracts"], errors="coerce")
    df_live["volume"] = pd.to_numeric(df_live["volume"], errors="coerce")

    live_agg = (
        df_live.groupby(["strike_price", "contract_type"])
        .agg({"oi_contracts": "sum", "volume": "sum"})
        .reset_index()
    )

    calls_live = live_agg[live_agg["contract_type"] == "call_options"][
        ["strike_price", "oi_contracts", "volume"]
    ].rename(columns={"oi_contracts": "Call OI", "volume": "Call Volume"})

    puts_live = live_agg[live_agg["contract_type"] == "put_options"][
        ["strike_price", "oi_contracts", "volume"]
    ].rename(columns={"oi_contracts": "Put OI", "volume": "Put Volume"})

    live_agg = pd.merge(calls_live, puts_live, on="strike_price", how="outer").fillna(0)

    # ---------- CSV T1 ----------
    csv_t1 = df[df["timestamp"] == t1].groupby("strike_price")[
        ["call_oi", "put_oi", "call_vol", "put_vol"]
    ].sum().reset_index()

    csv_t1 = csv_t1.rename(columns={
        "call_oi": "Call OI",
        "put_oi": "Put OI",
        "call_vol": "Call Volume",
        "put_vol": "Put Volume",
    })
    live_agg["strike_price"] = pd.to_numeric(live_agg["strike_price"], errors="coerce")
    csv_t1["strike_price"] = pd.to_numeric(csv_t1["strike_price"], errors="coerce")
    if not csv_mode:
        # Live OI
        oi_base = live_agg.copy()
    else:
        # T1 OI
        oi_base = csv_t1.copy()

    oi_range = oi_base[
        (oi_base["strike_price"] >= low) &
        (oi_base["strike_price"] <= high)
    ]
    call_sum = (oi_range["Call OI"] * oi_range["strike_price"]).sum()
    put_sum = (oi_range["Put OI"] * oi_range["strike_price"]).sum()
    diff_sum = put_sum - call_sum



    # ---------- OI / VOLUME DELTA (TOGGLE CONTROLLED) ----------
    if not csv_mode:
        # Live vs T1
        merged_oi = pd.merge(
            live_agg,
            csv_t1,
            on="strike_price",
            how="outer",
            suffixes=("_new", "_old")
        ).fillna(0)
    
    else:
        # T1 vs T2
        csv_t2 = df[df["timestamp"] == t2].groupby("strike_price")[
            ["call_oi", "put_oi", "call_vol", "put_vol"]
        ].sum().reset_index()
    
        csv_t2 = csv_t2.rename(columns={
            "call_oi": "Call OI",
            "put_oi": "Put OI",
            "call_vol": "Call Volume",
            "put_vol": "Put Volume",
        })
    
        merged_oi = pd.merge(
            csv_t1,
            csv_t2,
            on="strike_price",
            how="outer",
            suffixes=("_new", "_old")
        ).fillna(0)
    
    delta_live = pd.DataFrame({
      
        "strike_price": merged_oi["strike_price"],
        "Δ Call OI": merged_oi["Call OI_new"] - merged_oi["Call OI_old"],
        "Δ Put OI": merged_oi["Put OI_new"] - merged_oi["Put OI_old"],
        "Δ Call Volume": merged_oi["Call Volume_new"] - merged_oi["Call Volume_old"],
        "Δ Put Volume": merged_oi["Put Volume_new"] - merged_oi["Put Volume_old"],
    })
    delta_range = delta_live[
        (delta_live["strike_price"] >= low) &
        (delta_live["strike_price"] <= high)
    ]
    
    d_call_sum = (delta_range["Δ Call OI"] * delta_range["strike_price"]).sum()
    d_put_sum = (delta_range["Δ Put OI"] * delta_range["strike_price"]).sum()
    d_diff = d_put_sum - d_call_sum



    # ---------- LIVE MAX PAIN ----------
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
            (-sum(A[i:] * B[i:]) + G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
             - sum(M[:i] * L[:i]) + sum(G[i:] * L[i:]) - G[i] * sum(L[i:])) / 10000
            for i in range(len(df))
        ]
        return df[["strike_price", "Current"]]

    live_mp = compute_max_pain(mp)
    now_ts = get_ist_hhmm()
    live_mp.columns = ["strike_price", f"MP ({now_ts})"]

    # ---------- FINAL MERGE ----------
    final = (
        merged
        .merge(live_mp, on="strike_price", how="left")
        .merge(delta_live, on="strike_price", how="left")
    )

    final["△ MP 1"] = final[f"MP ({now_ts})"] - final[f"MP ({t1})"]
 

    final = final[
        [
            "strike_price",
            f"MP ({now_ts})",
            f"MP ({t1})",
            f"MP ({t2})",
            "△ MP 1",
            "△ MP 2",
            "Δ Call OI",
            "Δ Put OI",
            "Δ Call Volume",
            "Δ Put Volume",
        ]
    ].round(0).astype("Int64")

    # ---------- HIGHLIGHTING ----------
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


    # Scaling factor based on asset
    scale = 100_000_000 if UNDERLYING == "BTC" else 1_000_000
    
    summary_rows.append([
        f"{UNDERLYING} OI × Strike",
        int(call_sum / scale),
        int(put_sum / scale),
        int(diff_sum / scale),
    ])
    
    summary_rows.append([
        f"{UNDERLYING} ΔOI × Strike",
        int(d_call_sum / scale),
        int(d_put_sum / scale),
        int(d_diff / scale),
    ])



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



summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "Type",
        "Σ Call OI × Strike",
        "Σ Put OI × Strike",
        "Put − Call"
    ]
)

summary_placeholder.subheader("📊 OI Weighted Summary (Compact View)")
summary_placeholder.dataframe(summary_df, use_container_width=True)



# -------------------------------------------------
# PCR TABLE
# -------------------------------------------------
pcr_df = pd.DataFrame(
    pcr_rows,
    columns=["Asset", "PCR OI (Current)", "PCR OI (T1)", "PCR OI (T2)"]
).set_index("Asset")

st.subheader("📊 PCR Snapshot")
st.dataframe(pcr_df.round(3), use_container_width=True)


st.divider()
st.subheader("⚡ Live Delta Snapshot (Collector Logic)")

# ---------- BTC ----------
st.markdown("### 🟠 BTC")

btc_live = fetch_live_collector_data("BTC", selected_expiry)

if btc_live.empty:
    st.warning("BTC live data unavailable")
else:
    btc_live = compute_max_pain_collector(btc_live)

    btc_cols = [
        "strike_price",

        "call_mark", "call_oi", "call_volume",
        "call_gamma", "call_delta", "call_vega",

        "put_mark", "put_oi", "put_volume",
        "put_gamma", "put_delta", "put_vega",

        "max_pain",
        "timestamp_IST",
        "Expiry",
    ]

    st.dataframe(
        btc_live[btc_cols],
        use_container_width=True
    )

st.divider()

# ---------- ETH ----------
st.markdown("### 🟣 ETH")

eth_live = fetch_live_collector_data("ETH", selected_expiry)

if eth_live.empty:
    st.warning("ETH live data unavailable")
else:
    eth_live = compute_max_pain_collector(eth_live)

    eth_cols = [
        "strike_price",

        "call_mark", "call_oi", "call_volume",
        "call_gamma", "call_delta", "call_vega",

        "put_mark", "put_oi", "put_volume",
        "put_gamma", "put_delta", "put_vega",

        "max_pain",
        "timestamp_IST",
        "Expiry",
    ]

    st.dataframe(
        eth_live[eth_cols],
        use_container_width=True
    )
