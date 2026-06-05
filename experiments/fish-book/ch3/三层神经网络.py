import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def init_network():
    network = {
        "W1": np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        "b1": np.array([0.1, 0.2, 0.3]),
        "W2": np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        "b2": np.array([0.1, 0.2]),
        "W3": np.array([[0.1, 0.3], [0.2, 0.4]]),
        "b3": np.array([0.1, 0.2]),
    }
    return network


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def better_softmax(a, C):
    exp_a = np.exp(a + np.log(C))
    sum_exp_a = np.sum(exp_a + np.log(C))


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    print("网络参数初始化完成")
    print("W1 =", W1)
    print("b1 =", b1)
    print("W2 =", W2)
    print("b2 =", b2)
    print("W3 =", W3)
    print("b3 =", b3)
    print()

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    print("第一层")
    print("a1 =", a1)
    print("z1 =", z1)
    print()

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    print("第二层")
    print("a2 =", a2)
    print("z2 =", z2)
    print()

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    print("第三层")
    print("a3 =", a3)
    print("y =", y)

    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print()
    print("最终输出 =", y)
