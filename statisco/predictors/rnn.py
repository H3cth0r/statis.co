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
    def train(self, df, column_name, n_epochs=2000, batch_size=8):
        timeseries = df[[column_name]].values.astype("float32")
        # scaler = MinMaxScaler()
        # scaler.fit(timeseries)
        # timeseries = scaler.transform(timeseries)
        train_size = int(len(timeseries)*0.67)
        test_size = len(timeseries) - train_size
        train, test = timeseries[:train_size], timeseries[train_size:]

        X_train, y_train = create_dataset(train, lookback=self.lookback)
        X_test, y_test = create_dataset(test, lookback=self.lookback)


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
          for batch in range(0, X_train.shape[0]):
            X_b = Tensor(X_train[batch:batch+batch_size])
            if X_b.shape[0] != batch_size: continue
            y_b = Tensor(y_train[batch:batch+batch_size])
            loss = train_step(X_b, y_b)
          if epoch % 100 == 0:
              print(f"epoch: {epoch}, loss: {loss.numpy()}, rmse: {np.sqrt(loss.numpy())}")

    def predict(self, data, future=4): return self.model(Tensor(data, future)).numpy()[0]
