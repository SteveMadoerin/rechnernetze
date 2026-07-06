import os
from datetime import datetime

import yfinance as yf

# Target folder
OUT_DIR = r"E:\Golden Predict Agent\HistoricalData\NVDA"
os.makedirs(OUT_DIR, exist_ok=True)

ticker = "NVDA"

# --- 1-minute intraday data (Yahoo only serves the last ~30 days for 1m) ---
# intraday = yf.download(
#     ticker,
#     start="2026-06-12",
#     end="2026-06-20",
#     interval="1m",
# )
intraday = yf.download(
    ticker,
    start="2026-06-04",
    end="2026-06-12",
    interval="1m",
)

if intraday.empty:
    print("No intraday data returned (check the date range / interval limits).")
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intraday_path = os.path.join(OUT_DIR, f"{ticker}_1m_{stamp}.csv")
    intraday.to_csv(intraday_path)
    print(f"Saved intraday data -> {intraday_path}  ({len(intraday)} rows)")

# --- Full daily history (optional, comment out if not needed) ---
daily = yf.Ticker(ticker).history(period="max")
daily_path = os.path.join(OUT_DIR, f"{ticker}_daily_max.csv")
daily.to_csv(daily_path)
print(f"Saved daily history -> {daily_path}  ({len(daily)} rows)")
