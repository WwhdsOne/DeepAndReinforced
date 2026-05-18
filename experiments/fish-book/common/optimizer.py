import numpy as np


class Adam:
    """Adam 优化器

    参数：
        lr:      学习率，默认 0.001
        beta1:   一阶矩估计衰减率，默认 0.9
        beta2:   二阶矩估计衰减率，默认 0.999
        eps:     防止除零的小量，默认 1e-8
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}    # 一阶矩估计（梯度的指数移动平均）
        self.v = {}    # 二阶矩估计（梯度平方的指数移动平均）
        self.t = 0     # 迭代次数

    def change_lr(self, lr):
        self.lr = lr

    def update(self, params, grads):
        """执行一次参数更新。

        Args:
            params: 参数字典 {"W1": np.array, "b1": np.array, ...}
            grads:  梯度字典，结构与 params 相同
        """
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # 一阶矩估计：mt = beta1 * mt-1 + (1 - beta1) * gt
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # 二阶矩估计：vt = beta2 * vt-1 + (1 - beta2) * gt^2
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 参数更新
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
