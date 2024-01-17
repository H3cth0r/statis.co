from tinygrad import Tensor
from tinygrad.jit import TiniJit

class LSTMCell:
    def __init__(self, input_size, hidden_size, dropout):
        self.dropout        = dropout

        self.weights_ih     = Tensor.uniform(hidden_size * 4, input_size)
        self.bias_ih        = Tensor.uniform(hidden_size * 4)
        self.weights_hh     = Tensor.uniform(hidden_size * 4, hidden_size)
        self.bias_hh        = Tensor.uniform(hidden_size * 4)
    def __call__(self, x, hc):
        gates               = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)

        i, f, g, o          = gates.chunk(4, 1)
        i, f, g, o          = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

        c                   = (f * hc[x.shape[0]:]) + (i * g)
        h                   = (o * c.tanh()).dropout(self.dropout)
        return Tensor.cat(h, c).realize()


class LSTM:
    def __init__(self, input_size, hidden_size, layers, dropout):
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.layers         = layers

        self.cells = [LSTMCell(input_size, hidden_size, dropout) if i == 0 else LSTMCell(hidden_size, hidden_size, dropout if i != layers - 1 else 0) for i in range(layers)]

    def __call__(self, x, hc):
        @TiniJit
        def _do_step(x_, hc_):
            return self.do_step(x_, hc_)

        if hc is None:
            hc              = Tensor.zeros(self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)

        output              = None
        for t in range(x.shape[0]):
            hc              = _do_step(x[t] + 1 - 1, hc)
            if output is None:
                output      = hc[-1:, :x.shape[1]]
            else:
                output      = output.cat(hc[-1:, :x.shape[1]], dim=0).realize()
        return output, hc

    def do_step(self, x, hc):
        new_hc              = [x]
        for i, cell in enumerate(self.cells):
            new_hc.append(cell(new_hc[i][:x.shape[0]], hc[i]))
        return Tensor.stack(new_hc[1:]).realize()

if __name__ == "__main__":
