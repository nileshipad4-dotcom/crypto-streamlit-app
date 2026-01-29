import streamlit as st
import pandas as pd
import requests
import calendar
from datetime import datetime, timedelta, date
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Live Max Pain — Current + Next 5 Friday Expiries")

# -------------------------------------------------
# AUTO REFRESH (60s)
# -------------------------------------------------
if st_autorefresh(interval=60_000, key="auto_refresh"):
    st.cache_data.clear()

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
API_BASE = "https://api.india.delta.exchange/v2/tickers"
ASSETS = ["BTC", "ETH"]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_spot_price(symbol):
    r = requests.get(API_BASE, timeout=10).json()["result"]
    return float(next(x for x in r if x["symbol"] == f"{symbol}USD")["mark_price"])

def get_expiry_list():
    """Current day expiry + next 5 Fridays"""
    ist_today = get_ist_datetime().date()
    expiries = [ist_today]

    d = ist_today
    while len(expiries) < 6:
        d += timedelta(days=1)
        if d.weekday() == calendar.FRIDAY:
            expiries.append(d)

    return [e.strftime("%d-%m-%Y") for e in expiries]

def compute_max_pain(df):
    calls = df[df["contract_type"] == "call_options"]
    puts = df[df["contract_type"] == "put_options"]

    mp = pd.merge(
        calls[["strike_price", "mark_price", "oi_contracts"]]
            .rename(columns={"mark_price": "call_mark", "oi_contracts": "call_oi"}),
        puts[["strike_price", "mark_price", "oi_contracts"]]
            .rename(columns={"mark_price": "put_mark", "oi_contracts": "put_oi"}),
        on="strike_price",
        how="outer"
    ).fillna(0).sort_values("strike_price")

    pain = []
    strikes = mp["strike_price"].values

    for i, strike in enumerate(strikes):
        call_loss = (mp["call_oi"] * (mp["strike_price"] - strike).clip(lower=0)).sum()
        put_loss = (mp["put_oi"] * (strike - mp["strike_price"]).clip(lower=0)).sum()
        pain.append(call_loss + put_loss)

    mp["pain"] = pain
    return mp.loc[mp["pain"].idxmin(), "strike_price"]

# -------------------------------------------------
# MAIN
# -------------------------------------------------
expiries = get_expiry_list()

for asset in ASSETS:
    spot = get_spot_price(asset)

    rows = []

    for expiry in expiries:
        try:
            df = pd.json_normalize(
                requests.get(
                    f"{API_BASE}?contract_types=call_options,put_options"
                    f"&underlying_asset_symbols={asset}"
                    f"&expiry_date={expiry}",
                    timeout=20
                ).json()["result"]
            )
        except Exception:
            continue

        if df.empty:
            continue

        for c in ["strike_price", "mark_price", "oi_contracts"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        mp = compute_max_pain(df)

        rows.append({
            "Expiry": expiry,
            "Max Pain": int(mp),
            "Spot": int(spot),
            "MP − Spot": int(mp - spot),
        })

    result = pd.DataFrame(rows)

    st.subheader(f"{asset} — Live Max Pain")
    st.dataframe(
        result.style.apply(
            lambda r: ["background-color:#8B0000;color:white"] * len(r)
            if abs(r["MP − Spot"]) == result["MP − Spot"].abs().min()
            else ["" for _ in r],
            axis=1
        ),
        use_container_width=True
    )
