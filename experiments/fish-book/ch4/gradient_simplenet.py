import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def cross_entropy_error(y, t):
    """
    交叉熵误差。
    支持：
    - y, t 为一维向量
    - y, t 为二维批量输入
    - t 为 one-hot 编码或标签索引
    """
    y = np.asarray(y)
    t = np.asarray(t)
    delta = 1e-7

    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1) if t.ndim != 0 else np.array([t])

    if t.ndim == 1 and y.shape[0] == 1 and t.size == y.shape[1]:
        t = t.reshape(1, -1)

    batch_size = y.shape[0]

    if t.ndim == 1 or (t.ndim == 2 and t.shape == y.shape):
        if t.ndim == 2 and t.shape == y.shape:
            return -1 * np.sum(t * np.log(y + delta)) / batch_size
        return -1 * np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

    return -1 * np.sum(t * np.log(y + delta)) / batch_size


def numerical_gradient(func, x: np.ndarray):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = func(x)

        x[idx] = tmp_val - h
        fxh2 = func(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()
    return grad


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


def f(W):
    return net.loss(x, t)


if __name__ == '__main__':
    net = simpleNet()
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    learning_rate = 0.1
    steps = 100

    for i in range(steps):
        f = lambda W: net.loss(x, t)
        dW = numerical_gradient(f, net.W)
        net.W -= learning_rate * dW

        if i % 10 == 0:
            print(f"step {i}")
            print("loss =", net.loss(x, t))
            print("result =", net.predict(x))
            print("ans =", np.argmax(net.predict(x)))
            print("W = ", net.W)
            print()
