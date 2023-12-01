import numpy as np
from statisco.statistics import closingReturns

def test_closingReturns_1():
    stock_data  = np.random.rand(10000)
    print('generated inside')
    returns     = closingReturns(stock_data)
    print('end returns')


if __name__ == "__main__":
    test_closingReturns_1()
