from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad import nn

import numpy as np

from ..tinygrad.GRU import GRUCell, GRUModel
from ..tinygrad.loss import MSELoss 
from ..preprocessing.normalization import MinMaxScaler


def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return np.array(X).squeeze(), np.array(y).squeeze()

class GRU:
    def __init__(self, lookback=4):
        self.model = GRUModel()
        self.optimizer = nn.optim.Adam(nn.state.get_parameters(self.model), lr=0.001)
        self.lookback = lookback
    def train(self, df, column_name, n_epochs=2000, batch_size=8, plot_results=False):
        self.timeseries = df[[column_name]].values.astype("float32")
        train_size = int(len(self.timeseries)*0.67)
        test_size = len(self.timeseries) - train_size
        train, test = self.timeseries[:train_size], self.timeseries[train_size:]

        self.X_train, self.y_train = create_dataset(train, lookback=self.lookback)
        self.X_test, self.y_test = create_dataset(test, lookback=self.lookback)

        @TinyJit
        def train_step(X_batch, y_batch):
          with Tensor.train():
            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = MSELoss(preds, y_batch)
            loss.backward()
            self.optimizer.step()
            return loss
        
        for epoch in range(n_epochs):
          loss = 0
          for batch in range(0, self.X_train.shape[0]):
            X_b = Tensor(self.X_train[batch:batch+batch_size])
            if X_b.shape[0] != batch_size: continue
            y_b = Tensor(self.y_train[batch:batch+batch_size])
            loss = train_step(X_b, y_b)
          if epoch % 100 == 0 or epoch==n_epochs-1:
              print(f"epoch: {epoch}, loss: {loss.numpy()}, rmse: {np.sqrt(loss.numpy())}")

        if plot_results:
            train_plot = np.ones_like(self.timeseries) * np.nan
            y_pred = self.model(Tensor(self.X_train))
            y_pred = y_pred[:, -1]
            train_plot[self.lookback:train_size] = self.model(Tensor(self.X_train))[:, -1].reshape(-1, 1).numpy()
            test_plot = np.ones_like(self.timeseries) * np.nan
            test_plot[train_size+self.lookback:len(self.timeseries)] = self.model(Tensor(self.X_test))[:, -1].reshape(-1, 1).numpy()
            return self.timeseries, train_plot, test_plot

    def predict_from_train(self, index=-1, future=4): return self.model(Tensor(self.X_train[[index]]), future).numpy()[0]
    def predict_new_data(self, data, future):return self.model(Tensor(data), future).numpy()[0]
    def predict(self, index=None, data=None, future=4):
        if index is not None: return self.predict_from_train(index, future)
        elif data is not None: return self.predict_new_data(data, future)
        else: raise ValueError("Either 'index' or 'data' parameter must be provided for prediction.")
    def save(self, filename="GRUModel.safetensors"): nn.state.safe_save(nn.state.get_state_dict(self.model), filename)
    def load(self, filename="GRUModel.safetensors"): nn.state.load_state_dict(self.model, nn.state.safe_load(filename))
