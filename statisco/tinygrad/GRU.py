from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad import dtypes
import math

class GRUCell:
  def __init__(self, input_size, hidden_size, dropout, c_type):
    self.dropout = dropout
    bound = math.sqrt(1 / hidden_size)
    self.weights_ih = Tensor.uniform(hidden_size*3, input_size,  low=-bound, high=bound, dtype=c_type)
    self.weights_hh = Tensor.uniform(hidden_size*3, hidden_size, low=-bound, high=bound, dtype=c_type)
    self.bias_ih = Tensor.zeros(hidden_size*3, dtype=c_type)
    self.bias_hh = Tensor.zeros(hidden_size*3, dtype=c_type)

  def __call__(self, x, h_prev):
    gates = x.linear(self.weights_ih.T, self.bias_ih) + h_prev.linear(self.weights_hh.T, self.bias_hh)
    r, z, n = gates.chunk(3, 1)
    r, z, n = r.sigmoid(), z.sigmoid(), n.tanh()
    h_candidate = n * r
    h = (1 - z) * h_prev + z * h_candidate
    h = h.dropout(self.dropout)
    return h

class GRUModel:
  def __init__(self, n_hidden=30, n_cells=3):
    self.n_hidden   = n_hidden
    self.n_cells    = n_cells
    self.gru_cells  = [GRUCell(input_size=1 if cell == 0 else n_hidden, hidden_size=n_hidden, dropout=0.3, c_type=dtypes.float) for cell in range(n_cells)]
    self.fc         = nn.Linear(self.n_hidden, 1)
  def __call__(self, x, future=0):
    outputs = []
    n_samples = x.shape[0]

    hts = [Tensor.zeros(n_samples, self.n_hidden, dtype=dtypes.float) for _ in range(self.n_cells)]
    for input_t in x.split(1, dim=1):
      for i in range(self.n_cells): hts[i] = self.gru_cells[i](input_t if i == 0 else hts[i-1], hts[i])
      output = self.fc(hts[-1])
      outputs.append(output)
    for f in range(future):
      for i in range(self.n_cells): hts[i] = self.gru_cells[i](outputs[-1] if i == 0 else hts[i-1], hts[i])
      output = self.fc(hts[-1])
      outputs.append(output)
    outputs = Tensor.stack(outputs, dim=1)
    outputs = outputs.squeeze(2)
    return outputs
