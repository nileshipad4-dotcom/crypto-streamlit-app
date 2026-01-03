# collector.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_DELTA = "https://api.india.delta.exchange/v2/tickers"
HEADERS = {"Accept": "application/json"}

EXPIRIES = ["03-01-2026"]
UNDERLYINGS = ["BTC", "ETH"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TIMESTAMPS_PATH = os.path.join(DATA_DIR, "timestamps.csv")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_to_numeric(val):
    try:
        return pd.to_numeric(val)
    except Exception:
        return val


def get_ist_time_HHMM():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    return ist_now.strftime("%H:%M")


# -------------------------------------------------
# API FETCH
# -------------------------------------------------
def fetch_single_expiry(underlying, expiry_date):

    url = (
        f"{API_DELTA}"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry_date}"
    )

    r = requests.get(url, headers=HEADERS, timeout=20)
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

    for col in [
        "strike_price", "mark_price",
        "oi_contracts", "volume",
        "gamma", "delta", "vega"
    ]:
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
            "call_mark", "call_oi", "call_volume",
            "call_gamma", "call_delta", "call_vega",
            "strike_price",
            "put_gamma", "put_delta", "put_vega",
            "put_volume", "put_oi", "put_mark"
        ]
    ]

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
    max_pain = []

    for i in range(n):
        q = -sum(A[i:] * B[i:])
        r = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        s = -sum(M[:i] * L[:i])
        t = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        max_pain.append(round((q + r + s + t) / 10000))

    df["max_pain"] = max_pain
    return df


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    ts = get_ist_time_HHMM()

    # ---- write timestamps.csv (append-only) ----
    ts_df = pd.DataFrame({"timestamp": [ts]})
    ts_df.to_csv(
        TIMESTAMPS_PATH,
        mode="a",
        header=not os.path.exists(TIMESTAMPS_PATH),
        index=False
    )

    for underlying in UNDERLYINGS:

        df = fetch_single_expiry(underlying, EXPIRIES[0])
        if df.empty:
            continue

        df["timestamp_IST"] = ts
        df = compute_max_pain(df)

        out_path = os.path.join(DATA_DIR, f"{underlying}.csv")

        df.to_csv(
            out_path,
            mode="a",
            header=not os.path.exists(out_path),
            index=False
        )

        print(f"Saved {underlying} snapshot @ {ts}")

    print(f"Timestamp logged @ {ts}")


if __name__ == "__main__":
    main()
