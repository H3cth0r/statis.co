import numpy as np
from statisco.statistics import closingReturns
import memray

def test_closingReturns_1():
    stock_data  = np.random.rand(10000)
    print('generated inside')
    returns     = closingReturns(stock_data)
    print('end returns')


if __name__ == "__main__":
    with memray.Tracker("output_tracker_test.bin", native_traces=True, trace_python_allocators=True):
        print("started")
        test_closingReturns_1()
        print("ended")

