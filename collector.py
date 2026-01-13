# collector.py (AUTO EXPIRY VERSION)

import requests
import pandas as pd
from datetime import datetime, timedelta, date
import calendar
import os

headers = {"Accept": "application/json"}

UNDERLYINGS = ["BTC", "ETH"]
API_DELTA = "https://api.india.delta.exchange/v2/tickers"


# -------------------------------------------------
# IST TIME HELPERS
# -------------------------------------------------
def get_ist_datetime():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_ist_time_HHMM():
    return get_ist_datetime().strftime("%H:%M")


# -------------------------------------------------
# EXPIRY LOGIC (SAME AS CODE 1)
# -------------------------------------------------
def get_expiries():
    ist_now = get_ist_datetime()
    today = ist_now.date()

    if ist_now.time() <= datetime.strptime("17:30", "%H:%M").time():
        latest_valid = today
    else:
        latest_valid = today + timedelta(days=1)

    expiries = set()
    expiries.add(latest_valid)

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
        lf = max(week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0)
        expiries.add(date(y, m, lf))

    return sorted(
        [d.strftime("%d-%m-%Y") for d in expiries],
        key=lambda x: datetime.strptime(x, "%d-%m-%Y"),
    )


# -------------------------------------------------
# DATA FETCH
# -------------------------------------------------
def safe_to_numeric(val):
    try:
        return pd.to_numeric(val)
    except:
        return val


def fetch_single_expiry(underlying, expiry_date):

    url = (
        f"{API_DELTA}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry_date}"
    )

    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        return pd.DataFrame()

    raw = r.json().get("result", [])
    df = pd.json_normalize(raw)
    if df.empty:
        return pd.DataFrame()

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

    for col in ["strike_price", "mark_price", "oi_contracts", "volume", "gamma", "delta", "vega"]:
        df[col] = safe_to_numeric(df[col])

    calls = df[df["contract_type"] == "call_options"].copy()
    puts = df[df["contract_type"] == "put_options"].copy()

    calls = calls.rename(columns={
        "mark_price": "call_mark",
        "oi_contracts": "call_oi",
        "volume": "call_volume",
        "gamma": "call_gamma",
        "delta": "call_delta",
        "vega": "call_vega"
    }).drop(columns=["contract_type"])

    puts = puts.rename(columns={
        "mark_price": "put_mark",
        "oi_contracts": "put_oi",
        "volume": "put_volume",
        "gamma": "put_gamma",
        "delta": "put_delta",
        "vega": "put_vega"
    }).drop(columns=["contract_type"])

    merged = pd.merge(calls, puts, on="strike_price", how="outer")

    merged = merged[
        [
            "call_mark","call_oi","call_volume",
            "call_gamma","call_delta","call_vega",
            "strike_price",
            "put_gamma","put_delta","put_vega",
            "put_volume","put_oi","put_mark"
        ]
    ]

    merged["Expiry"] = expiry_date
    merged = merged.sort_values("strike_price").reset_index(drop=True)

    return merged


# -------------------------------------------------
# MAX PAIN
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


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    ts = get_ist_time_HHMM()
    os.makedirs("data", exist_ok=True)

    expiries = get_expiries()
    selected_expiry = expiries[0]   # Nearest valid expiry

    print(f"Using Expiry: {selected_expiry}")

    for underlying in UNDERLYINGS:

        df = fetch_single_expiry(underlying, selected_expiry)
        if df.empty:
            print(f"No data for {underlying} {selected_expiry}")
            continue

        df["timestamp_IST"] = ts
        df = compute_max_pain(df)

        # 🔴 EXPIRY-SPECIFIC FILE
        out_path = f"data/{underlying}_{selected_expiry}.csv"

        file_exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        
        df.to_csv(
            out_path,
            mode="a",
            header=not file_exists,
            index=False
        )


        print(f"Saved {underlying} @ {ts} | Expiry {selected_expiry}")



if __name__ == "__main__":
    main()

