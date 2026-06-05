class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        """
        乘法层的反向传播。
        根据链式法则，计算损失函数对输入 x 和 y 的梯度。

        :param dout: 指的是从下一层（后一层）传过来的梯度，通常代表损失函数对该层输出的偏导数（dL/dout）。
        :return: dx, dy 分别表示损失函数对输入 x 和 y 的偏导数（dL/dx, dL/dy），将传递给上一层。
        """
        # 乘法层的反向传播特点是“互换乘数”：
        # 因为前向传播是 out = x * y
        # 所以 dL/dx = dout * y
        dx = dout * self.y

        # dL/dy = dout * x
        dy = dout * self.x

        # 将计算好的梯度返回，供上一层（前一层）的节点继续反向传播
        return dx, dy


if __name__ == "__main__":
    apple_origin_price = 100
    apple_tax = 1.1
    apple_num = 2
    dprice = 1
    a = MultiLayer()
    b = MultiLayer()
    A = a.forward(apple_origin_price, apple_tax)
    B = b.forward(apple_num, A)
    print("Price = ", B)

    print()

    # 从输出层（总价）开始反向传播，计算损失对数量和含税价格的梯度
    dapple_price, dapple_num = b.backward(dprice)
    # 继续反向传播，计算损失对原价和税率的梯度
    dapple, dtax = a.backward(dapple_num)

    # 打印各个梯度值，用于验证反向传播的正确性
    print("dapple_price = ", dapple_price)  # 损失对含税价格的梯度
    print("dapple_num = ", dapple_num)  # 损失对数量的梯度
    print("dapple = ", dapple)  # 损失对原价的梯度
    print("dtax = ", dtax)  # 损失对税率的梯度
