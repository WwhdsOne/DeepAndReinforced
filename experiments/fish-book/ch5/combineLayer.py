from common.multiLayer import MultiLayer
from common.addLayer import AddLayer

if __name__ == '__main__':
    apple_origin_price = 100
    apple_tax = 1.1
    apple_num = 2
    orange_origin_price = 150
    orange_tax = 1.2
    orange_num = 3
    dprice = 1
    a = MultiLayer()
    b = MultiLayer()
    c = MultiLayer()
    d = MultiLayer()
    e = AddLayer()
    A = a.forward(apple_origin_price, apple_tax)
    B = b.forward(apple_num, A)
    C = c.forward(orange_origin_price, orange_tax)
    D = d.forward(orange_num, C)
    E = e.forward(B, D)

    print("Price = ",E)

    print()

    # 从输出层（总价）开始反向传播，计算损失对数量和含税价格的梯度
    dapple_price, dapple_num = b.backward(dprice)
    # 继续反向传播，计算损失对原价和税率的梯度
    dapple, dtax = a.backward(dapple_num)

    # 打印各个梯度值，用于验证反向传播的正确性
    print("dapple_price = ", dapple_price)  # 损失对含税价格的梯度
    print("dapple_num = ", dapple_num)      # 损失对数量的梯度
    print("dapple = ", dapple)              # 损失对原价的梯度
    print("dtax = ", dtax)                  # 损失对税率的梯度

    # 从输出层（总价）开始反向传播，计算损失对数量和含税价格的梯度
    dorange_price, dorange_num = d.backward(dprice)
    # 继续反向传播，计算损失对原价和税率的梯度
    dorange, dtax = c.backward(dorange_num)
    print("dorange_price = ", dorange_price)
    print("dorange_num = ", dorange_num)
    print("dorange = ", dorange)
    print("dtax = ", dtax)
    print("dapple_price + dorange_price = ", dapple_price + dorange_price)
    print("dapple_num + dorange_num = ", dapple_num + dorange_num)
    print("dapple + dorange = ", dapple + dorange)
    print("dtax = ", dtax)