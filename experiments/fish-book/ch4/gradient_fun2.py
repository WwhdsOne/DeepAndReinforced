import numpy as np
import matplotlib.pylab as plt


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(func, x: np.ndarray):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = func(x)

        x[idx] = tmp_val - h
        fxh2 = func(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
    x -= lr * grad
    return x


if __name__ == '__main__':
    x = np.arange(-2.0, 2.1, 0.2)
    y = np.arange(-2.0, 2.1, 0.2)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    grad_x = np.zeros_like(X)
    grad_y = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            grad = numerical_gradient(function_2, np.array([X[i, j], Y[i, j]]))
            grad_x[i, j] = grad[0]
            grad_y[i, j] = grad[1]

    plt.figure(figsize=(7, 6))
    plt.quiver(
        X,
        Y,
        -grad_x,
        -grad_y,
        angles='xy',
        scale_units='xy',
        scale=20,
        color='gray',
        width=0.0035,
    )
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'$x_1$')
    plt.title(r'$f(x_0, x_1) = x_0^2 + x_1^2$ 的梯度')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
