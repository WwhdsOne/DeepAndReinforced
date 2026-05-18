"""TwoLayerNet: 基于误差反向传播的手写数字识别。"""
import sys
import os

sys.path.append(os.pardir)

import numpy as np
from collections import OrderedDict
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {"W1": weight_init_std * np.random.randn(input_size, hidden_size),
                       "b1": np.zeros(hidden_size),
                       "W2": weight_init_std * np.random.randn(hidden_size, output_size),
                       "b2": np.zeros(output_size)}

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for key in ("W1", "b1", "W2", "b2"):
            grads[key] = numerical_gradient(loss_W, self.params[key])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {"W1": self.layers["Affine1"].dW,
                 "b1": self.layers["Affine1"].db,
                 "W2": self.layers["Affine2"].dW,
                 "b2": self.layers["Affine2"].db}
        return grads


def load_mnist(data_dir: str) -> tuple:
    """加载 MNIST 数据集（使用 npz 格式）。"""
    import urllib.request
    import ssl
    from pathlib import Path

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "mnist.npz"

    if not cache.exists():
        urls = [
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            "https://github.com/fgnt/mnist/raw/master/mnist.npz",
        ]
        ctx = ssl._create_unverified_context()
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                    cache.write_bytes(resp.read())
                break
            except Exception:
                continue
        else:
            raise RuntimeError("MNIST 下载失败，请手动放置 mnist.npz")

    with np.load(cache) as data:
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def trainAndTest():
    # 超参数
    input_size = 784
    hidden_size = 256
    output_size = 10
    iters_num = 1000
    train_size = 60000
    batch_size = 128
    learning_rate = 0.001

    # 加载数据
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    x_train, t_train, x_test, t_test = load_mnist(data_dir)

    # 归一化并反色 (MNIST 风格)
    x_train = x_train.astype(np.float32).reshape(-1, 784) / 255.0
    x_test = x_test.astype(np.float32).reshape(-1, 784) / 255.0

    # 构建网络
    network = TwoLayerNet(input_size, hidden_size, output_size)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 每个 epoch 的迭代数
    iter_per_epoch = max(train_size // batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(
                f"iter {i:5d} | loss {loss:.4f} | "
                f"train acc {train_acc:.4f} | test acc {test_acc:.4f}"
            )

    print(f"\n最终训练准确率: {train_acc_list[-1]:.4f}")
    print(f"最终测试准确率: {test_acc_list[-1]:.4f}")

    return train_acc_list[-1], test_acc_list[-1]


if __name__ == "__main__":
    tmp_avg_train_acc, tmp_avg_test_acc = 0, 0
    for i in range(10):
        x, y = trainAndTest()
        tmp_avg_train_acc += x
        tmp_avg_test_acc += y
    print(f"平均训练准确率: {tmp_avg_train_acc / 10:.4f}")
    print(f"平均测试准确率: {tmp_avg_test_acc / 10:.4f}")
