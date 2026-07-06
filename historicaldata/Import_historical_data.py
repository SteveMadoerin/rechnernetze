import yfinance as yf

nvda = yf.Ticker("NVDA")
print(nvda)
# get stock info
print(nvda.info)

# get historical market data
print(nvda.history(period="max"))

# show actions (dividends, splits)
print(nvda.actions)

print(nvda.history(period="1d", interval="1m"))

yf.download("NVDA", start="2026-06-12", end="2026-06-20", interval="1m")

