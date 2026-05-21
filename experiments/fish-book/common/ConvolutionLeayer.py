from common.utils import im2col, col2im
import numpy as np


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        # 保存中间变量用于反向传播
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 保存输入和中间结果
        self.x = x
        col = im2col(x, FH, FW, self.stride, self.pad)
        self.col = col

        col_W = self.W.reshape(FN, -1).T  # (C*FH*FW, FN)
        self.col_W = col_W

        out = np.dot(col, col_W) + self.b  # (N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        """
        dout: 上游梯度，形状 (N, FN, out_h, out_w)
        返回: dx, dW, db
        """
        FN, C, FH, FW = self.W.shape

        # 1. 将 dout 转换为与 forward 中 out 相同的形状
        dout = dout.transpose(0, 2, 3, 1)  # (N, out_h, out_w, FN)
        dout = dout.reshape(-1, FN)  # (N*out_h*out_w, FN)

        # 2. 计算 db (偏置的梯度)
        db = np.sum(dout, axis=0)  # (FN,)

        # 3. 计算 dW (权重的梯度)
        # dout: (N*out_h*out_w, FN)
        # col:  (N*out_h*out_w, C*FH*FW)
        # dW = col.T @ dout -> (C*FH*FW, FN)
        dW = np.dot(self.col.T, dout)  # (C*FH*FW, FN)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 4. 计算 dx (输入的梯度)
        # dcol = dout @ col_W.T -> (N*out_h*out_w, C*FH*FW)
        dcol = np.dot(dout, self.col_W.T)

        # 使用 col2im 将 dcol 转换回图像形状
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        self.db = db
        self.dW = dW
        return dx, dW, db
