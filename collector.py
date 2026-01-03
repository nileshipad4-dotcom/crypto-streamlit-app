# collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

headers = {"Accept": "application/json"}

EXPIRIES = ["03-01-2026"]
UNDERLYINGS = ["BTC", "ETH"]

API_DELTA = "https://api.india.delta.exchange/v2/tickers"

def safe_to_numeric(val):
    try:
        return pd.to_numeric(val)
    except:
        return val

def fetch_single_expiry(underlying, expiry_date):

    url = (
        "https://api.india.delta.exchange/v2/tickers"
        f"?contract_types=call_options,put_options"
        f"&underlying_asset_symbols={underlying}"
        f"&expiry_date={expiry_date}"
    )
    r = requests.get(url, headers=headers)
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


def get_ist_time_HHMM():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    return ist_now.strftime("%H:%M")


def compute_max_pain(df):

    A = df["call_mark"].fillna(0).values
    B = df["call_oi"].fillna(0).values
    G = df["strike_price"].fillna(0).values
    L = df["put_oi"].fillna(0).values
    M = df["put_mark"].fillna(0).values

    n = len(df)

    Q = []
    R = []
    S = []
    T = []
    U = []

    for i in range(n):

        # Q = -SUMPRODUCT(A[i:], B[i:])
        q_val = -sum(A[i:] * B[i:])
        Q.append(q_val)

        # R = (G[i] * SUM(B[:i])) - SUMPRODUCT(G[:i], B[:i])
        r_val = G[i] * sum(B[:i]) - sum(G[:i] * B[:i])
        R.append(r_val)

        # S = -SUMPRODUCT(M[:i], L[:i])
        s_val = -sum(M[:i] * L[:i])
        S.append(s_val)

        # T = SUMPRODUCT(G[i:], L[i:]) - G[i] * SUM(L[i:])
        t_val = sum(G[i:] * L[i:]) - G[i] * sum(L[i:])
        T.append(t_val)

        # U = ROUND((Q + R + S + T) / 10000)
        U.append(round((q_val + r_val + s_val + t_val) / 10000))

    df["Q_call_cost"] = Q
    df["R_call_intrinsic"] = R
    df["S_put_cost"] = S
    df["T_put_intrinsic"] = T
    df["max_pain"] = U

    return df


def main():

    ts = get_ist_time_HHMM()
    os.makedirs("data", exist_ok=True)

    for underlying in UNDERLYINGS:

        df = fetch_single_expiry(underlying, EXPIRIES[0])
        if df.empty:
            continue

        df["timestamp_IST"] = ts

        # calculate max pain for THIS time group
        df = compute_max_pain(df)

        out_path = f"data/{underlying}.csv"

        df.to_csv(
            out_path,
            mode="a",
            header=not os.path.exists(out_path),
            index=False
        )

        print(f"Saved {underlying} data @ {ts}")


if __name__ == "__main__":
    main()





