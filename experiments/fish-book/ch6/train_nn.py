"""TwoLayerNet: 基于误差反向传播的手写数字识别优化版。"""

import os
import sys

from common.utils import load_mnist

sys.path.append(os.pardir)

import numpy as np
from collections import OrderedDict
from common.layers import Affine, Relu, SoftmaxWithLoss, BatchNorm, Adam, Dropout


class MultiLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {
            "W1": kaiming(input_size, hidden_size),  # Kaiming 初始化
            "b1": np.zeros(hidden_size),
            "gamma1": np.ones(hidden_size),  # BatchNorm 缩放参数
            "beta1": np.zeros(hidden_size),  # BatchNorm 平移参数
            "W2": xavier(hidden_size, output_size),  # Xavier 随机初始化
            "b2": np.zeros(output_size),
            "gamma2": np.ones(output_size),
            "beta2": np.zeros(output_size),
        }

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["BatchNorm1"] = BatchNorm(
            self.params["gamma1"], self.params["beta1"]
        )
        self.layers["Dropout1"] = Dropout(0.1)
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["BatchNorm2"] = BatchNorm(
            self.params["gamma2"], self.params["beta2"]
        )
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, BatchNorm):
                x = layer.forward(x, train_flg=train_flg)
            elif isinstance(layer, Dropout):
                x = layer.forward(x, train_flg=train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, lambda_reg=1e-4):
        """交叉熵损失 + L2 正则化"""
        y = self.predict(x, train_flg=True)
        ce_loss = self.lastLayer.forward(y, t)

        # L2 正则化：只对权重 W1、W2 施加惩罚
        l2_loss = lambda_reg * (
            np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)
        )
        return ce_loss + l2_loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

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
            "gamma1": self.layers["BatchNorm1"].dgamma,
            "beta1": self.layers["BatchNorm1"].dbeta,
            "W2": self.layers["Affine2"].dW,
            "b2": self.layers["Affine2"].db,
            "gamma2": self.layers["BatchNorm2"].dgamma,
            "beta2": self.layers["BatchNorm2"].dbeta,
        }
        return grads


def kaiming(input_size, hidden_size):
    return np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)


def xavier(input_size, hidden_size):
    return np.random.randn(input_size, hidden_size) * np.sqrt(
        2 / (input_size + hidden_size)
    )


def trainAndTest():
    # 超参数
    input_size = 784
    hidden_size = 256
    output_size = 10
    iters_num = 100
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
    network = MultiLayerNet(input_size, hidden_size, output_size)

    # Adam 优化器
    optimizer = Adam(lr=learning_rate)

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
        optimizer.update(network.params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            optimizer.change_lr(learning_rate * 0.9)
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
    loop_count = 1
    for i in range(loop_count):
        x, y = trainAndTest()
        tmp_avg_train_acc += x
        tmp_avg_test_acc += y
    print(f"平均训练准确率: {tmp_avg_train_acc / loop_count:.4f}")
    print(f"平均测试准确率: {tmp_avg_test_acc / loop_count:.4f}")
