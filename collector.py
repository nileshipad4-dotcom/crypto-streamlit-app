# collector.py
import requests
import pandas as pd
from datetime import datetime
import os

API_DELTA = "https://api.india.delta.exchange/v2/tickers"

def fetch_price_binance(symbol):
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        r = requests.get(url, params={"symbol": symbol}, timeout=10).json()
        return float(r["price"])
    except:
        return None

def fetch_option_chain(underlying, expiry):
    url = f"{API_DELTA}?contract_types=call_options,put_options&underlying_asset_symbols={underlying}&expiry_date={expiry}"
    r = requests.get(url, timeout=20).json()
    return pd.json_normalize(r.get("result", []))

EXPIRY = "11-12-2025"

def main():
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    btc = fetch_price_binance("BTCUSDT")
    eth = fetch_price_binance("ETHUSDT")

    btc_chain = fetch_option_chain("BTC", EXPIRY)
    eth_chain = fetch_option_chain("ETH", EXPIRY)

    # Add timestamp to each row
    btc_chain["timestamp"] = ts
    eth_chain["timestamp"] = ts

    # Create data folder if not exist
    os.makedirs("data", exist_ok=True)

    # Append to CSV (creates if not exist)
    btc_chain.to_csv("data/btc_chain.csv", mode="a", header=not os.path.exists("data/btc_chain.csv"), index=False)
    eth_chain.to_csv("data/eth_chain.csv", mode="a", header=not os.path.exists("data/eth_chain.csv"), index=False)

    # Save prices too
    with open("data/prices.csv", "a") as f:
        f.write(f"{ts},{btc},{eth}\n")

if __name__ == "__main__":
    main()
