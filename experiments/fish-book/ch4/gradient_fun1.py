import numpy as np
import matplotlib.pylab as plt


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


def d_function_1(x):
    return 0.02 * x + 0.1


if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
    y = function_1(x)

    x1, x2 = 5.0, 10.0
    y1, y2 = function_1(x1), function_1(x2)
    dy1, dy2 = d_function_1(x1), d_function_1(x2)

    tangent_x = np.arange(0.0, 20.0, 0.1)
    tangent_y1 = dy1 * (tangent_x - x1) + y1
    tangent_y2 = dy2 * (tangent_x - x2) + y2

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y, label="f(x)")
    plt.plot(
        tangent_x, tangent_y1, linestyle="--", color="red", label=f"tangent at x={x1}"
    )
    plt.plot(
        tangent_x, tangent_y2, linestyle="--", color="green", label=f"tangent at x={x2}"
    )
    plt.scatter([x1, x2], [y1, y2], color=["red", "green"])
    plt.legend()
    plt.show()
