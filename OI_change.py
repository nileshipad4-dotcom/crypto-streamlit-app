import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import base64
from io import StringIO
from streamlit_autorefresh import st_autorefresh

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

def get_available_expiries():
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

    return sorted(expiries, key=lambda x: datetime.strptime(x, "%d-%m-%Y"))


def get_next_expiries(selected_expiry, count=3):
    expiries = get_available_expiries()

    # Always include selected expiry
    result = [selected_expiry]

    # Add future expiries even if CSV doesn't exist yet
    try:
        base = datetime.strptime(selected_expiry, "%d-%m-%Y")
        while len(result) < count:
            base += timedelta(days=7)  # weekly expiry
            result.append(base.strftime("%d-%m-%Y"))
    except Exception:
        pass

    return result


    idx = expiries.index(selected_expiry)
    return expiries[idx : idx + count]


def read_csv_from_github(path):
    url = f"{GITHUB_API}/repos/{CRYPTO_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()

    content = base64.b64decode(r.json()["content"]).decode()
    return pd.read_csv(StringIO(content))

# =========================================================
# LOAD CLEAN OI HISTORY
# =========================================================

def load_data(symbol, expiry):
    df = read_csv_from_github(f"{RAW_DIR}/{symbol}_{expiry}_snapshots.csv")
    df["_row"] = range(len(df))

    raw = pd.to_datetime(df["timestamp_IST"], format="%H:%M")
    base = datetime(2000,1,1)
    out, last, day = [], None, 0

    for t in raw:
        if last is not None and t < last:
            day += 1
        out.append(base + timedelta(days=day, hours=t.hour, minutes=t.minute))
        last = t

    df["timestamp_IST"] = out
    
    # 🔑 CRITICAL FIX: remove duplicate strike rows per timestamp
    df = df.drop_duplicates(subset=["timestamp_IST", "strike_price"])
    
    return df

def get_latest_csv_time(symbol, expiry):
    try:
        df = read_csv_from_github(
            f"{RAW_DIR}/{symbol}_{expiry}_snapshots.csv"
        )
        if df.empty or "timestamp_IST" not in df:
            return "—"

        raw = pd.to_datetime(
            df["timestamp_IST"],
            format="%H:%M",
            errors="coerce"
        ).dropna()

        if raw.empty:
            return "—"

        base = datetime(2000, 1, 1)
        out, last, day = [], None, 0

        for t in raw:
            if last is not None and t < last:
                day += 1
            out.append(
                base + timedelta(days=day, hours=t.hour, minutes=t.minute)
            )
            last = t

        return max(out).strftime("%H:%M")

    except Exception:
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



# =========================================================
# WINDOW ENGINE
# =========================================================

def build_all_windows(df, gap):
    times = (
        df.sort_values("_row")
          .drop_duplicates("timestamp_IST")["timestamp_IST"]
          .tolist()
    )

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
    rows=[]
    windows = build_all_windows(df,gap)

    for t1,t2 in windows:
        r = build_row(df,t1,t2)
        if r: rows.append(r)

    if windows:
        r = build_row(df, windows[-1][1], df["timestamp_IST"].max(), live=True)
        if r: rows.append(r)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("_end").drop(columns="_end")
    return out


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
        f"&expiry_date={expiry}"
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
    m["max_pain"]=0

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


def append_raw(path, df):
    url = f"{GITHUB_API}/repos/{CRYPTO_REPO}/contents/{path}"
    headers={"Authorization":f"token {GITHUB_TOKEN}"}
    r=requests.get(url,headers=headers)
    sha,old=None,""

    if r.status_code==200:
        sha=r.json()["sha"]
        old=base64.b64decode(r.json()["content"]).decode()

    new=df.to_csv(index=False,header=not bool(old))
    payload={
        "message":"raw snapshot",
        "content":base64.b64encode((old+new).encode()).decode(),
        "branch":GITHUB_BRANCH
    }
    if sha: payload["sha"]=sha
    requests.put(url,headers=headers,json=payload)

# =========================================================
# UI
# =========================================================

st.title("BTC & ETH OI Change Scanner")

btc_p = get_delta_price("BTC")
eth_p = get_delta_price("ETH")

c1,c2 = st.columns(2)
c1.metric("BTC Price", f"{btc_p:,.2f}" if btc_p else "—")
c2.metric("ETH Price", f"{eth_p:,.2f}" if eth_p else "—")

c_exp, c_gap, c_thr = st.columns([2,1,1])

with c_exp:
    expiries = get_available_expiries()
    expiry = st.selectbox(
        "Select Expiry",
        expiries,
        index=len(expiries) - 1
    )

with c_gap:
    gap = st.selectbox("Min Gap (minutes)", [5,10,15,20,30,45,60], index=2)





for sym in ["BTC", "ETH"]:
    st.subheader(sym)
    # Per-symbol default threshold


    df = process_windows(load_data(sym, expiry), gap)

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
        times=df["TIME"].tolist()
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

# =========================================================
# RAW PUSH
# =========================================================



bucket = ((datetime.utcnow().hour * 60) + datetime.utcnow().minute) // 4
col_t, col_c = st.columns([1, 2])

with col_t:
    st.toggle("📤 Push raw snapshots", key="push_enabled")

bucket, remaining = get_bucket_and_remaining()
mm, ss = divmod(remaining, 60)

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

if (
    st.session_state.push_enabled
    and 120 < remaining < 180
    and st.session_state.last_push_bucket != bucket
):
    expiries_to_collect = get_next_expiries(expiry, count=3)
    
    for sym in ["BTC", "ETH"]:
        for exp in expiries_to_collect:
            df = fetch_live(sym, exp)
            if not df.empty:
                append_raw(
                    f"{RAW_DIR}/{sym}_{exp}_snapshots.csv",
                    df
                )


    st.session_state.last_push_bucket = bucket
    st.success("Raw snapshots pushed successfully.")
