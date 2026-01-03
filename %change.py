import os
import requests
import pandas as pd
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================
BASE_URL = "https://api.binance.com"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================
# HELPERS
# ==================================================
def get_all_symbols():
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    data = requests.get(url, timeout=10).json()
    return [
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]

def get_5m_change(symbol):
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 6
    }
    data = requests.get(url, params=params, timeout=10).json()

    if len(data) < 6:
        return None

    price_5m_ago = float(data[0][4])
    current_price = float(data[-1][4])

    return round(((current_price - price_5m_ago) / price_5m_ago) * 100, 4)

# ==================================================
# MAIN
# ==================================================
symbols = get_all_symbols()
rows = []

for sym in symbols:
    try:
        change = get_5m_change(sym)
        if change is not None:
            rows.append({
                "Symbol": sym,
                "Change_5m_%": change
            })
    except Exception:
        pass

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No data fetched")

df = df.sort_values("Change_5m_%", ascending=False)

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
file_path = f"{DATA_DIR}/crypto_5m_change_{timestamp}.csv"

df.to_csv(file_path, index=False)

print(f"CSV saved at: {file_path}")
