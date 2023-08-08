import yfinance as yf

spy = yf.Ticker("SPY")

print(spy.info)