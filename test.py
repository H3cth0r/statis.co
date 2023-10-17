import yfinance as yf
import numpy as np
import pandas as pd
import time
from statisco.processingFunctions import closingReturns, averageReturns, varianceReturns, stdDeviation, covarianceReturns, correlationReturns, compoundInterest, moneyMadeInAYear, compoundInterestTime, calculateSMA, calculateEMA, calculateWMA, calculateATR, calculateATRwma
import statisco.processingFunctions as stco
import math

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
def covarianceReturns_NP(x, y):
    xy = x*y
    return np.mean(xy) - (np.mean(x) * np.mean(y))
def correlationReturns_NP(xyCovar_t, xVar, yVar):
    return xyCovar_t / (math.sqrt(xVar) * math.sqrt(yVar))
def compoundInterest_NP(P, r, t):
    return P * (1 + r)**t
def moneyMadeInAYear_NP(P, r, t):
    return compoundInterest_NP(P, r, t) * r
def compoundInterestTime_NP(r):
    return -np.log(r)/np.log(1 + r)
def calculateSMA_NP(closings, window_t):
    result = np.zeros_like(closings)
    for i in range(window_t - 1, len(closings)):
        result[i] = np.mean(closings[i - window_t+1 : i +1])
    return result 
def calculateEMA_NP(returns, SMA, window_t):
    size = len(returns)
    result = np.zeros(size, dtype=np.float64)
    multiplier = 2.0 / (window_t + 1)
    first = True

    for i in range(window_t - 1, size):
        if first:
            result[i] = returns[i] * multiplier + SMA[i] * (1 - multiplier)
            first = False
        else:
            prevValue = result[i - 1]
            result[i] = returns[i] * multiplier + prevValue * (1 - multiplier)

    return result
def calculateWMA_NP(returns, window_t):
    WMA     = np.zeros(returns)
    for i in range(window_t, len(returns) + 1):
        sum = 0
        counter = 1
        for j in range(i-window_t, i):
            sum     += returns[j] (counter / window_t)
            counter += 1
        WMA[i] = sum
    return WMA
def calculateATR_NP(Close_t, High_t, Low_t, window_t):
    size = len(Close_t)
    trueRange = np.zeros(size, dtype=np.double)
    ATR = np.zeros(size, dtype=np.double)

    for i in range(size):
        if i > 0:
            prev_i = i - 1
            trueRange[i] = max(
                abs(High_t[i] - Low_t[i]),
                max(
                    abs(High_t[i] - Close_t[prev_i]),
                    abs(Low_t[i] - Close_t[prev_i])
                )
            )
        else:
            trueRange[i] = abs(High_t[i] - Low_t[i])

    sum = 0
    for i in range(window_t, size):
        sum = 0
        for j in range(i - window_t, i):
            sum += trueRange[j]
        ATR[i] = sum

    return ATR
def calculateATRwma_NP(CLose_t, High_t, Low_t, window_t):
    size = len(Close_t)
    true_range = np.zeros(size)
    weighted_atr = np.zeros(size)
    for i in range(size):
        if i > 0:
            true_range[i] = max(
                abs(High_t[i] - Low_t[i]),
                max(
                    abs(High_t[i] - Close_t[i - 1]),
                    abs(Low_t[i] - Close_t[i - 1])
                )
            )
        else:
            true_range[i] = abs(High_t[i] - Low_t[i])
    
    for i in range(window_t, size):
        weighted_sum = 0
        weight_sum = 0
        weight = 0
        for j in range(i - window_t, i + 1):
            weight += j - i + window_t + 1
            weighted_sum += true_range[j] * weight
            weight_sum += weight

        weighted_atr[i] = weighted_sum / weight_sum
    return weighted_atr
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0  
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


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

def test_covarianceReturns_1():
    nvda            = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    amd             = yf.download("AMD", start="2022-01-01", end="2022-12-31")
    adjClose_nvda   = nvda["Adj Close"].to_numpy()
    adjClose_amd    = amd["Adj Close"].to_numpy()

    returns_nvda    = closingReturns(adjClose_nvda)
    returns_amd     = closingReturns(adjClose_amd)

    print(f"len: {len(returns_nvda)}")

    start_time          = time.time()
    covr                = covarianceReturns(returns_nvda, returns_amd)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    std_np = covarianceReturns_NP(returns_nvda, returns_amd)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns_nvda.shape}")
    print(covr)
    print(std_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
def test_covarianceReturns_2():
    nvda            = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    amd             = yf.download("AMD", start="2022-01-01", end="2022-12-31")
    adjClose_nvda   = np.random.uniform(nvda["Adj Close"].min(), nvda["Adj Close"].max(), size=(20000,))
    adjClose_amd    = np.random.uniform(amd["Adj Close"].min(), amd["Adj Close"].max(), size=(20000,))

    returns_nvda    = closingReturns(adjClose_nvda)
    returns_amd     = closingReturns(adjClose_amd)

    print(f"len: {len(returns_nvda)}")

    start_time          = time.time()
    covr                = covarianceReturns(returns_nvda, returns_amd)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time = time.time()
    std_np = covarianceReturns_NP(returns_nvda, returns_amd)
    end_time = time.time()
    nptimes = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns_nvda.shape}")
    print(covr)
    print(std_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
def test_correlationReturns_1():
    nvda            = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    amd             = yf.download("AMD", start="2022-01-01", end="2022-12-31")
    adjClose_nvda   = nvda["Adj Close"].to_numpy()
    adjClose_amd    = amd["Adj Close"].to_numpy()

    returns_nvda    = closingReturns(adjClose_nvda)
    returns_amd     = closingReturns(adjClose_amd)

    covr            = covarianceReturns(returns_nvda, returns_amd)

    avgReturns_nvda = averageReturns(returns_nvda)
    avgReturns_amd  = averageReturns(returns_amd)


    start_time          = time.time()
    corr                = correlationReturns(covr, avgReturns_nvda, avgReturns_amd)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time          = time.time()
    std_np              = correlationReturns_NP(covr, avgReturns_nvda, avgReturns_amd)
    end_time            = time.time()
    nptimes             = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns_nvda.shape}")
    print(corr)
    print(std_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
def test_correlationReturns_2():
    nvda            = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    amd             = yf.download("AMD", start="2022-01-01", end="2022-12-31")
    adjClose_nvda   = np.random.uniform(nvda["Adj Close"].min(), nvda["Adj Close"].max(), size=(20000,))
    adjClose_amd    = np.random.uniform(amd["Adj Close"].min(), amd["Adj Close"].max(), size=(20000,))

    returns_nvda    = closingReturns(adjClose_nvda)
    returns_amd     = closingReturns(adjClose_amd)

    covr            = covarianceReturns(returns_nvda, returns_amd)

    avgReturns_nvda = averageReturns(returns_nvda)
    avgReturns_amd  = averageReturns(returns_amd)


    start_time          = time.time()
    corr                = correlationReturns(covr, avgReturns_nvda, avgReturns_amd)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time          = time.time()
    std_np              = correlationReturns_NP(covr, avgReturns_nvda, avgReturns_amd)
    end_time            = time.time()
    nptimes             = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns_nvda.shape}")
    print(corr)
    print(std_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")

def test_compoundInterest():
    start_time          = time.time()
    corr                = compoundInterest(2, 3, 2)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time          = time.time()
    std_np              = compoundInterest_NP(2, 3, 2)
    end_time            = time.time()
    nptimes             = f"Numpy time: \t\t{end_time - start_time :.10f}"
    print(corr)
    print(std_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
def test_moneyMadeInAYear():
    print(f"NP: \t{moneyMadeInAYear_NP(2,3,2)}")
    print(f"C:  \t{moneyMadeInAYear(2,3,2)}")
def test_compoundInterestTime():
    print(f"NP: \t{compoundInterestTime_NP(2)}")
    print(f"C:  \t{compoundInterestTime(2)}")
def test_SMA_1():
    closings = np.random.uniform(100.0, 150.0, 100)
    start_time          = time.time()
    cc  = calculateSMA(closings, 5)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time          = time.time()
    npy = calculateSMA_NP(closings, 5)
    end_time            = time.time()
    nptimes             = f"Numpy time: \t\t{end_time - start_time :.10f}"
    print(npy)
    print(cc)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
    print("similarity: ", cosine_similarity(npy, cc))
def test_SMA_2():
    closings = np.random.uniform(100.0, 250.0, 2000)
    start_time          = time.time()
    cc  = calculateSMA(closings, 5)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time          = time.time()
    npy = calculateSMA_NP(closings, 5)
    end_time            = time.time()
    nptimes             = f"Numpy time: \t\t{end_time - start_time :.10f}"
    print(npy)
    print(cc)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
    print("similarity: ", cosine_similarity(npy, cc))
def test_EMA_1():
    closings            = np.random.uniform(100.0, 150.0, 100)
    SMA                 = calculateSMA(closings, 5)
    start_time          = time.time()
    cc                  = calculateEMA(closings, SMA, 5)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time          = time.time()
    npy                 = calculateEMA_NP(closings, SMA,5)
    end_time            = time.time()
    nptimes             = f"Numpy time: \t\t{end_time - start_time :.10f}"
    print(cc)
    print(npy)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
    print("similarity: ", cosine_similarity(npy, cc))
def test_EMA_2():
    closings            = np.random.uniform(100.0, 150.0, 10000)
    SMA                 = calculateSMA(closings, 5)
    start_time          = time.time()
    cc                  = calculateEMA(closings, SMA, 5)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time          = time.time()
    npy                 = calculateEMA_NP(closings, SMA,5)
    end_time            = time.time()
    nptimes             = f"Numpy time: \t\t{end_time - start_time :.10f}"
    print(cc)
    print(npy)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
    print("similarity: ", cosine_similarity(npy, cc))
def test_WMA_1():
    stock_data          = yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    adjClose            = stock_data["Adj Close"].to_numpy()

    returns             = closingReturns(adjClose)

    print(f"len: {len(returns)}")

    start_time          = time.time()
    WMA                 = calculateWMA(returns, 7)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time      = time.time()
    WMA_np          = calculateWMA_NP(returns, 7)
    end_time        = time.time()
    nptimes         = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(WMA)
    print(WMA_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
def test_ATR_1():
    # CLose, high low, window_t
    stock_data          =   yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    High                =   stock_data["High"].to_numpy()
    Close               =   stock_data["Close"].to_numpy()
    Low                 =   stock_data["Low"].to_numpy()

    start_time          = time.time()
    ATR                 = calculateATR(High, Close, Low, 7)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time      = time.time()
    ATR_np          = calculateATR_NP(High, Close, Low, 7)
    end_time        = time.time()
    nptimes         = f"Numpy time: \t\t{end_time - start_time :.10f}"

    # print(f"returns: {returns.shape}")
    print(ATR)
    print(ATR_np)
    print(ctimes)
    print(nptimes)
    print("C wins" if ctimes < nptimes else "numpy wins")
def test_ATRwma_1():
    # CLose, high low, window_t
    stock_data          =   yf.download("NVDA", start="2022-01-01", end="2022-12-31")
    High                =   stock_data["High"].to_numpy()
    Close               =   stock_data["Close"].to_numpy()
    Low                 =   stock_data["Low"].to_numpy()

    start_time          = time.time()
    ATR                 = calculateATRwma(High, Close, Low, 7)
    end_time            = time.time()
    ctimes              = f"C extension time: \t{end_time - start_time :.10f}"

    start_time      = time.time()
    ATR_np          = calculateATRwma_NP(High, Close, Low, 7)
    end_time        = time.time()
    nptimes         = f"Numpy time: \t\t{end_time - start_time :.10f}"

    print(f"returns: {returns.shape}")
    print(ATR)
    print(ATR_np)
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
    print("="*60)
    test_covarianceReturns_1()
    print("="*60)
    print("COVARIANCE TEST")
    test_covarianceReturns_2()
    print("="*60)
    test_correlationReturns_1()
    print("="*60)
    print("CORRELATION TEST")
    test_correlationReturns_2()
    print("="*60)
    test_compoundInterest()
    print("="*60)
    test_moneyMadeInAYear()
    print("="*60)
    test_compoundInterestTime()
    print("="*60)
    test_SMA_1()
    print("="*60)
    test_SMA_2()
    print("="*60)
    test_EMA_1()
    print("="*60)
    test_EMA_2()
    print("="*60)
    print("ATR")
    test_ATR_1()
    # # print("="*60)
    # print("WMA ATR")
    # test_ATRwma_1()
