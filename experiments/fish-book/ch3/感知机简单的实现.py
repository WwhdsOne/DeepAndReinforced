import numpy as np
import matplotlib.pyplot as plt


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    # NumPy 1.20 以后不再支持 np.int，直接使用内置 int 即可。
    return np.array(x > 0, dtype=int)


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y_step = step_function(x)
    y_sigmoid = sigmoid(x)
    y_relu = np.maximum(x, 0)

    plt.plot(x, y_step, label="step function", linestyle="--")
    plt.plot(x, y_sigmoid, label="sigmoid")
    plt.plot(x, y_relu, label="relu")
    plt.ylim(-0.1, 1.1)  # 指定y轴的范围
    plt.title("Step Function vs Sigmoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
