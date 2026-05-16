import numpy as np


class Softmax:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = _softmax(x)
        self.loss = _cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = _softmax(x)
        self.loss = _cross_entropy_error(self.y, t)
        # 将标签转为 one-hot，供 backward 使用
        if t.ndim == 1 or t.size != self.y.size:
            self.t = np.eye(self.y.shape[1])[t.reshape(-1)]
        else:
            self.t = t
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


def _softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def _cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7

    if t.size == y.size:
        return -1 * np.sum(t * np.log(y + delta)) / batch_size
    return -1 * np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
