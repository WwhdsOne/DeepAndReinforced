import os

import numpy as np
from common.layers import Convolution, Relu, SoftmaxWithLoss, Pooling, Affine
from collections import OrderedDict

from common.utils import load_mnist
import time


class SimpleConvNet:
    """一个简单的卷积神经网络。"""

    def __init__(
        self, input_dim=(1, 28, 28), conv_param=None, hidden_size=100, output_size=10
    ):
        if conv_param is None:
            conv_param = {"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1}
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        self.params = {}
        self.params["W1"] = np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        ) * np.sqrt(1.0 / (input_size * filter_size * filter_size))
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = np.random.randn(pool_output_size, hidden_size) * np.sqrt(
            1.0 / pool_output_size
        )
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = np.random.randn(hidden_size, output_size) * np.sqrt(
            1.0 / hidden_size
        )
        self.params["b3"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["pad"],
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])

        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])

        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        # forward
        loss = self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads, loss

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """交叉熵损失 + L2 正则化"""
        y = self.predict(x)
        ce_loss = self.lastLayer.forward(y, t)

        # L2 正则化：只对权重 W1、W2 施加惩罚
        # l2_loss = lambda_reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        return ce_loss

    def accuracy(self, x, t, batch_size=2000):
        """分批计算准确率，避免全量数据一次性前向传播导致内存爆炸。"""
        total_correct = 0
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i : i + batch_size]
            t_batch = t[i : i + batch_size]
            y = self.predict(x_batch)
            y = np.argmax(y, axis=1)
            if t_batch.ndim != 1:
                t_batch = np.argmax(t_batch, axis=1)
            total_correct += np.sum(y == t_batch)
        return total_correct / float(x.shape[0])


def trainAndTest():
    # 超参数
    input_dim = (1, 28, 28)
    hidden_size = 50
    output_size = 10
    iters_num = 10
    train_size = 60000
    batch_size = 128
    learning_rate = 0.001

    # 加载数据
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    x_train, t_train, x_test, t_test = load_mnist(data_dir)

    # 归一化并反色 (MNIST 风格)
    x_train = x_train.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    x_test = x_test.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0

    # 构建网络
    network = SimpleConvNet(
        input_dim=input_dim, hidden_size=hidden_size, output_size=output_size
    )

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 每个 epoch 的迭代数
    iter_per_epoch = max(train_size // batch_size, 1)

    start_time = time.time()
    for i in range(iters_num):
        # 在循环中添加进度提示
        elapsed = time.time() - start_time
        print(
            f"Iteration {i}/{iters_num} ({(i / iters_num) * 100:.1f}%) - {elapsed:.1f}s elapsed"
        )
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad, loss = network.gradient(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            network.params[key] -= learning_rate * grad[key]

        train_loss_list.append(loss)

        # 每 5 个 epoch 评估一次准确率（避免每轮评估全量数据集拖慢训练）
        eval_interval = iter_per_epoch * 5
        if i % eval_interval == 0 or i == iters_num - 1:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(
                f"iter {i:5d} | loss {loss:.4f} | "
                f"train acc {train_acc:.4f} | test acc {test_acc:.4f}"
            )

    # 最终评估
    final_train_acc = network.accuracy(x_train, t_train)
    final_test_acc = network.accuracy(x_test, t_test)
    print(f"\n最终训练准确率: {final_train_acc:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")

    return final_train_acc, final_test_acc


if __name__ == "__main__":
    tmp_avg_train_acc, tmp_avg_test_acc = 0, 0
    loop_count = 1
    for i in range(loop_count):
        x, y = trainAndTest()
        tmp_avg_train_acc += x
        tmp_avg_test_acc += y
    print(f"平均训练准确率: {tmp_avg_train_acc / loop_count:.4f}")
    print(f"平均测试准确率: {tmp_avg_test_acc / loop_count:.4f}")
