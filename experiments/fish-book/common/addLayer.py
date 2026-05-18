
class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        out = x + y
        return out
    def backward(self, dout):
        """
        加法层的反向传播。
        根据链式法则，计算损失函数对输入 x 和 y 的梯度。

        :param dout: 指的是从下一层（后一层）传过来的梯度，通常代表损失函数对该层输出的偏导数（dL/dout）。
        :return: dx, dy 分别表示损失函数对输入 x 和 y 的偏导数（dL/dx, dL/dy），将传递给上一层。
        """
        # 乘法层的反向传播特点是“互换乘数”：
        # 因为前向传播是 out = x + y
        # 所以 dL/dx = dout * 1
        dx = 1 * dout

        # dL/dy = dout * x
        dy = 1 * dout

        # 将计算好的梯度返回，供上一层（前一层）的节点继续反向传播
        return dx, dy