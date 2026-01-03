import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://api.binance.com"

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
    params = {"symbol": symbol, "interval": "1m", "limit": 6}
    data = requests.get(url, params=params, timeout=10).json()

    if len(data) < 6:
        return None

    p5 = float(data[0][4])
    p0 = float(data[-1][4])
    return round(((p0 - p5) / p5) * 100, 3)

symbols = get_all_symbols()
rows = []

for s in symbols:
    try:
        chg = get_5m_change(s)
        if chg is not None:
            rows.append({"Symbol": s, "Change_5m_%": chg})
    except Exception:
        pass

df = pd.DataFrame(rows).sort_values("Change_5m_%", ascending=False)

fname = f"crypto_5m_change_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(fname, index=False)

print(f"Saved {fname}")
