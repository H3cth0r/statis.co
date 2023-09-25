import yfinance as yf

if __name__ == "__main__":
    print("TEST")
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    print(stock_data)


