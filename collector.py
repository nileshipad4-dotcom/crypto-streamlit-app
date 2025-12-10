# collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ----------------------------------------
# CONFIG
# ----------------------------------------
headers = {"Accept": "application/json"}

EXPIRIES = ["11-12-2025"]
UNDERLYINGS = ["BTC", "ETH"]

API_DELTA = "https://api.india.delta.exchange/v2/tickers"


# ----------------------------------------
# SAFE NUMERIC
# ----------------------------------------
def safe_to_numeric(val):
    try:
        return pd.to_numeric(val)
    except:
        return val


# ----------------------------------------
# FETCH OPTION CHAIN IN REQUIRED TABLE FORMAT
# ----------------------------------------
def fetch_single_expiry(underlying, expiry_date):

    url = (
        "https://api.india.delta.exchange/v2/tickers"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry_date}"
    )

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"❌ Error fetching {underlying} {expiry_date}: {r.text}")
        return pd.DataFrame()

    raw = r.json().get("result", [])
    df = pd.json_normalize(raw)

    if df.empty:
        return pd.DataFrame()

    df = df[
        [
            "strike_price",
            "contract_type",
            "mark_price",
            "oi_contracts",
            "volume",
            "greeks.gamma",
            "greeks.delta",
            "greeks.vega",
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
            "call_mark", "call_oi", "call_volume",
            "call_gamma", "call_delta", "call_vega",
            "strike_price",
            "put_gamma", "put_delta", "put_vega",
            "put_volume", "put_oi", "put_mark"
        ]
    ]

    merged["Expiry"] = expiry_date
    return merged.sort_values("strike_price").reset_index(drop=True)


# ----------------------------------------
# GET IST TIME HH:MM ONLY
# ----------------------------------------
def get_ist_time_HHMM():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    return ist_now.strftime("%H:%M")


# ----------------------------------------
# MAIN DATA COLLECTION
# ----------------------------------------
def main():
    ts = get_ist_time_HHMM()

    os.makedirs("data", exist_ok=True)

    for underlying in UNDERLYINGS:

        df = fetch_single_expiry(underlying, EXPIRIES[0])
        if df.empty:
            print(f"❌ No data for {underlying}")
            continue

        # Add IST time column
        df["timestamp_IST"] = ts

        out_path = f"data/{underlying}.csv"

        df.to_csv(
            out_path,
            mode="a",
            header=not os.path.exists(out_path),
            index=False
        )

        print(f"✔ Saved {underlying} data @ {ts}")


if __name__ == "__main__":
    main()
