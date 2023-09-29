import yfinance as yf
import numpy as np
import pandas as pd
import time
from statisco.processingFunctions import closingReturns, averageReturns, varianceReturns, stdDeviation
import statisco.processingFunctions as stco

def closingReturns_NP(adjsc):
    returnsPerRow = np.zeros_like(adjsc, dtype=float)
    returnsPerRow[:-1] = adjsc[:-1] / adjsc[1:] - 1
    # returnsPerRow = np.append(returnsPerRow, 0)
    return returnsPerRow
def averageReturns_NP(returns):
    return returns.mean() 
def varianceReturns_NP(returns_t, averageReturns_t):
    diff_sqd = (returns_t - averageReturns_t) ** 2
    return np.mean(diff_sqd)
def test_closingReturns_1():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    print(stock_data.head())
    adjClose = stock_data["Adj Close"].to_numpy()

    start_time = time.time()
    returns = closingReturns(adjClose)
    end_time = time.time()
    rtimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    returns_np = closingReturns_NP(adjClose)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(returns_np.shape) 
    stock_data["NpReturns"] = returns_np
    stock_data["CReturns"] = returns

    print(stock_data.head())

    print(f"adjClose: {adjClose.shape}")
    print(f"returns: {returns.shape}")
    print(f"returns np: {returns_np.shape}")
    print(rtimes)
    print(nptimes)
    print("C wind" if rtimes > nptimes else "numpy wins")

def test_closingReturns_2():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose = np.random.uniform(stock_data["Adj Close"].min(), stock_data["Adj Close"].max(), size=(10000,))

    start_time = time.time()
    returns = closingReturns(adjClose)
    end_time = time.time()
    rtimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    returns_np = closingReturns_NP(adjClose)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"adjClose: {adjClose.shape}")
    print(f"returns: {returns.shape}")
    print(f"returns np: {returns_np.shape}")
    print(rtimes)
    print(nptimes)
    print("C wins" if rtimes < nptimes else "numpy wins")

def test_averageReturns_1():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose = stock_data["Adj Close"].to_numpy()

    returns = closingReturns(adjClose)

    start_time = time.time()
    avgReturns = averageReturns(returns)
    end_time = time.time()
    avgtimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    avgReturns_np = averageReturns_NP(returns)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(f"average avgReturns: {avgReturns}")
    print(f"average avgReturns_np: {avgReturns_np}")
    print(avgtimes)
    print(nptimes)
    print("C wins" if avgtimes < nptimes else "numpy wins")

def test_averageReturns_2():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose = np.random.uniform(stock_data["Adj Close"].min(), stock_data["Adj Close"].max(), size=(10000,))

    returns = closingReturns(adjClose)

    start_time = time.time()
    avgReturns = averageReturns(returns)
    end_time = time.time()
    avgtimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    avgReturns_np = averageReturns_NP(returns)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(f"average avgReturns: {avgReturns}")
    print(f"average avgReturns_np: {avgReturns_np}")
    print(avgtimes)
    print(nptimes)
    print("C wins" if avgtimes < nptimes else "numpy wins")

def test_varianceReturns_1():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose = stock_data["Adj Close"].to_numpy()

    returns = closingReturns(adjClose)
    avgReturns = averageReturns(returns)

    print("this is the returns type: ", type(returns))
    print("this is the type averageReturns: ", type(avgReturns))
    print(f"len: {len(returns)}")

    start_time = time.time()
    varReturns = varianceReturns(returns, avgReturns)
    end_time = time.time()
    ctimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    varReturns_np = varianceReturns_NP(returns, avgReturns)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(f"average avgReturns: {varReturns}")
    print(f"average avgReturns_np: {varReturns_np}")
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")

    
def test_varianceReturns_2():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose = np.random.uniform(stock_data["Adj Close"].min(), stock_data["Adj Close"].max(), size=(10000,))

    returns = closingReturns(adjClose)
    avgReturns = averageReturns(returns)

    print("this is the returns type: ", type(returns))
    print("this is the type averageReturns: ", type(avgReturns))
    print(f"len: {len(returns)}")

    start_time = time.time()
    varReturns = varianceReturns(returns, avgReturns)
    end_time = time.time()
    ctimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    varReturns_np = varianceReturns_NP(returns, avgReturns)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(f"average avgReturns: {varReturns}")
    print(f"average avgReturns_np: {varReturns_np}")
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")

def test_stdDeviation_1():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose = stock_data["Adj Close"].to_numpy()

    returns = closingReturns(adjClose)

    print(f"len: {len(returns)}")

    start_time = time.time()
    stdDev = stdDeviation(returns)
    end_time = time.time()
    ctimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    std_np = np.std(returns) 
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(stdDev)
    print(std_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")

def test_stdDeviation_2():
    stock_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose = np.random.uniform(stock_data["Adj Close"].min(), stock_data["Adj Close"].max(), size=(10000,))

    returns = closingReturns(adjClose)

    print(f"len: {len(returns)}")

    start_time = time.time()
    stdDev = stdDeviation(returns)
    end_time = time.time()
    ctimes = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    std_np = np.std(returns) 
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(stdDev)
    print(std_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")

if __name__ == "__main__":
    print("TEST")
    test_closingReturns_1()
    print("="*60)
    test_closingReturns_2()
    print("="*60)
    test_averageReturns_1()
    print("="*60)
    test_averageReturns_2()
    print("="*60)
    test_varianceReturns_1()
    print("="*60)
    test_varianceReturns_2()
    print("="*60)
    test_stdDeviation_1()
    print("="*60)
    test_stdDeviation_2()



