# Definition of some loss functions for tinygrad
def MSELoss(y_pred, y_true):return ((y_pred - y_true)**2).mean()
def MAELoss(y_pred, y_true):return ((y_pred - y_true).abs()).mean()
