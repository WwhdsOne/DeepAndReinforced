import numpy as np


class BatchNormalizationLayer:
    """批归一化层（Batch Normalization）

    前向传播：
        1. 计算 mini-batch 的均值 μ_B 和方差 σ_B²
        2. 标准化：x̂ = (x - μ_B) / √(σ_B² + ε)
        3. 缩放与平移：y = γ * x̂ + β

    反向传播沿计算图逐步回溯：out → x̂ → xc → x，同时计算 dγ 和 dβ。
    推导参考：https://arxiv.org/abs/1502.03167
    """

    def __init__(
        self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, eps=1e-7
    ):
        self.gamma = gamma  # 缩放参数 (D,)
        self.beta = beta  # 平移参数 (D,)
        self.momentum = momentum  # 移动平均的动量
        self.eps = eps  # 防止除零的小量

        # 推理时使用的移动平均均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # 反向传播时使用的中间变量
        self.batch_size = None
        self.xc = None  # x - μ（中心化后的输入）
        self.x_hat = None  # 标准化后的 x
        self.std = None  # √(σ_B² + ε)
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.batch_size = x.shape[0]

        # 初始化移动平均
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1:])
        if self.running_var is None:
            self.running_var = np.zeros(x.shape[1:])

        if train_flg:
            # 训练模式：使用当前 mini-batch 的统计量
            mu = np.mean(x, axis=0)  # 均值 μ_B
            self.xc = x - mu  # 中心化
            var = np.mean(self.xc**2, axis=0)  # 方差 σ_B²
            self.std = np.sqrt(var + self.eps)  # √(σ_B² + ε)
            self.x_hat = self.xc / self.std  # 标准化

            # 更新移动平均
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            # 推理模式：使用移动平均的统计量
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        N = self.batch_size

        # dγ 和 dβ
        self.dgamma = np.sum(dout * self.x_hat, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # 对 x̂ 的梯度
        dx_hat = dout * self.gamma

        # 沿计算图回溯：x̂ → xc → x
        dxc = dx_hat / self.std

        # std 依赖于 var，var 依赖于 xc
        dstd = -np.sum(dx_hat * self.xc / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / N) * self.xc * dvar

        # mu 依赖于 x，xc = x - mu
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / N

        return dx
