"""TwoLayerNet 与数据集加载 —— 供 ch5/ch6 复用。"""
import os
import sys
import urllib.request
import ssl
from pathlib import Path

import numpy as np
from collections import OrderedDict

from common.layers import Affine, Relu, SoftmaxWithLoss
from common.gradient import numerical_gradient


def kaiming_init(n_in, n_out):
    """He（Kaiming）初始化，适用于 ReLU 后的权重。"""
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)


def xavier_init(n_in, n_out):
    """Xavier（Glorot）初始化，适用于 Sigmoid/Tanh 后的权重。"""
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01, W1=None, W2=None):
        """
        两层神经网络。

        Parameters
        ----------
        weight_init_std : float
            默认权重初始化标准差（ch5 旧方式）。
        W1, W2 : ndarray, optional
            自定义权重矩阵。若传入则覆盖默认初始化（ch6 自定义初始化用）。
        """
        if W1 is None:
            W1 = weight_init_std * np.random.randn(input_size, hidden_size)
        if W2 is None:
            W2 = weight_init_std * np.random.randn(hidden_size, output_size)

        self.params = {
            "W1": W1,
            "b1": np.zeros(hidden_size),
            "W2": W2,
            "b2": np.zeros(output_size),
        }

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

        grads = {
            "W1": self.layers["Affine1"].dW,
            "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db,
        }
        return grads


def load_mnist(data_dir: str) -> tuple:
    """加载 MNIST 数据集（使用 npz 格式），若不存在则自动下载。"""
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
