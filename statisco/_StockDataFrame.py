import pandas
import yfinance as yf
import numpy as np
from .preprocessing.normalization import MinMaxScaler
from .statistics import closingReturns
from .indicators.MAs import SMA

from contextlib import redirect_stdout
import io
def run_function_silently(func):
    with io.StringIO() as fake_stdout:
        with redirect_stdout(fake_stdout):
            result = func()

        # Now, fake_stdout.getvalue() contains the suppressed print output
        return result, fake_stdout.getvalue()

class StockDataFrame(pandas.DataFrame):
    def __init__(self, data=None, ticker=None, *args, **kwargs):
        if isinstance(data, pandas.DataFrame):
            super(StockDataFrame, self).__init__(data, *args)
            return

        if isinstance(data, str) :
            downloaded_data = self.download(data, **kwargs)
        elif isinstance(ticker, str):
            downloaded_data = self.download(ticker, **kwargs)
        super(StockDataFrame, self).__init__(downloaded_data, *args)

    def calculate(self, close_returns=False, sma=True, interval=3):
        if close_returns:
            self["CloseReturns"] = closingReturns(self["Adj Close"])
        if sma:
            self["SMA"] = SMA(self["Close"], interval)
        return

    def download(self, ticker, start=None, end=None, interval="1d", *args, **kwargs):
        # param_list = inspect.getfullargspec(yf.download).args
        param_dict = {
            'tickers': ticker,
            'start': start,
            'end': end,
            'interval': interval
        }
        param_dict.update(kwargs)
        donwloaded, _ = run_function_silently(lambda: yf.download(**param_dict))
        return donwloaded
    def update(self):
        pass

    def normalize(self, inplace=False):
        data            = self.copy().to_numpy()
        data.astype(np.double)
        self.min_max_scaler  = MinMaxScaler()
        self.min_max_scaler.fit(data)
        if inplace: 
            self[:] = self.min_max_scaler.transform(data)
        else:
            return self.min_max_scaler.transform(data)

    def indicators(self):
        pass
