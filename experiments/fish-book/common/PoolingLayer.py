import numpy as np

from common.utils import im2col, col2im


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.x = None
        self.arg_max = None
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展开成列
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 🔥 关键：记住最大值的位置！
        self.arg_max = np.argmax(col, axis=1)  # 保存位置

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x  # 保存输入形状，backward要用
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        pool_size = self.pool_h * self.pool_w

        # 1. 把 dout 转换成和 forward 时 arg_max 一样的形状
        dout = dout.transpose(0, 2, 3, 1)  # (N,OH,OW,C)
        dout = dout.reshape(-1)  # 展平

        # 2. 构建一个全0矩阵，把梯度放到最大值的位置上
        dcol = np.zeros((dout.size, pool_size), dtype=np.float32)
        dcol[np.arange(dout.size), self.arg_max] = dout

        # 3. 还原形状 + col2im（🔥 修正这里）
        dcol = dcol.reshape(-1, C * pool_size)  # 可能需要调整
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
