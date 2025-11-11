import pandas
import yfinance as yf
import numpy as np
from .preprocessing.normalization import MinMaxScaler
from .statistics import closingReturns
from .indicators.MAs import SMA, EMA, WMA, MACD
from .indicators.ATRs import ATR

from .utils.api_utils import cache_response


from contextlib import redirect_stdout
import io
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json

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

        self.app = None

    def calculate(self, close_returns=False, sma=False, ema=False, wma=False, atr=False, interval=3, smooth=2):
        if close_returns:
            self["CloseReturns"] = closingReturns(self["Adj Close"])
        if sma:
            self["SMA"] = SMA(self["Close"], interval)
        if ema:
            self["EMA"] = EMA(self["Close"], SMA(self["Close"], interval), smooth, interval)
        if wma:
            self["WMA"] = WMA(self["Close"], interval)
        if atr: 
            self["ATR"] = ATR(self["Close"], self["High"], self["Low"], interval)
        return self

    def calculate_MACD(self, short_window=12, long_window=26, signal_window=9):
        self["MACD"], self["MACD_SignalLine"], self["MACD_Histogram"] = MACD(self["Close"], short_window, long_window, signal_window)
        return self

    def download(self, ticker, start=None, end=None, interval="1d", *args, **kwargs):
        # param_list = inspect.getfullargspec(yf.download).args
        param_dict = {
            'tickers': ticker,
            'start': start,
            'end': end,
            'interval': interval,
            'auto_adjust': False,
            'progress': False
        }
        param_dict.update(kwargs)
        downloaded, _ = run_function_silently(lambda: yf.download(**param_dict))

        if isinstance(downloaded.columns, pandas.MultiIndex):
            downloaded.columns = downloaded.columns.droplevel(1)
        return downloaded

    def update(self):
        pass

    def normalize(self, fit=True, transform=True, inplace=False, data=None):
        """
        Method for applying min max normalziation to the dataframe.
        This will store the fitted model, for new data.
        1. First fit .
        2. Then transform.
        """
        if fit:
            data            = self.copy().to_numpy()
            data.astype(np.double)
            self.min_max_scaler  = MinMaxScaler()
            self.min_max_scaler.fit(data)
        if transform:
            if inplace: 
                self[:] = self.min_max_scaler.transform(data)
            else:
                return self.min_max_scaler.transform(data)

    def indicators(self):
        pass
    
    def init_api(self):
        """
        Initialize FastAPI Application.
        """
        self.app = FastAPI()
        return self.app

    def add_endpoint(self, path: str, description: str):
        if not self.app:
            raise Exception("API not initialized. init_api() first.")

        @self.app.get(path, description=description)
        @cache_response # This decorator is assumed to be in your api_utils
        async def dynamic_endpoint(request: Request):
            try:
                filtered_df = self.copy()

                for key, value in request.query_params.items():
                    col = key
                    op = 'eq' 

                    if '__' in key:
                        col, op = key.split('__')
                    
                    if col not in filtered_df.columns:
                        continue 

                    try:
                        numeric_value = pandas.to_numeric(value)
                    except ValueError:
                        numeric_value = value

                    # Apply the filter based on the operator
                    if op == 'eq':
                        filtered_df = filtered_df[filtered_df[col] == numeric_value]
                    elif op == 'gt':
                        filtered_df = filtered_df[filtered_df[col] > numeric_value]
                    elif op == 'lt':
                        filtered_df = filtered_df[filtered_df[col] < numeric_value]
                    elif op == 'gte':
                        filtered_df = filtered_df[filtered_df[col] >= numeric_value]
                    elif op == 'lte':
                        filtered_df = filtered_df[filtered_df[col] <= numeric_value]

                return JSONResponse(content=json.loads(filtered_df.to_json(orient='split')))
            except Exception as e:
                print(e)
                return JSONResponse(content={"error": str(e)}, status_code=500)

    def run_api(self, host="127.0.0.1", port=8000):
        """
        Run FastAPI application using uvicorn
        """
        if not self.app:
            raise Exception("API not initilized. Call init_api() first.")
        uvicorn.run(self.app, host=host, port=port)
