import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from data_utils import get_mnist_loaders


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(784, 512)
        self.layer2 = torch.nn.Linear(512, 256)
        self.layer3 = torch.nn.Linear(256, 128)
        self.layer4 = torch.nn.Linear(128, 64)
        self.layer5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 输入 x 的形状为 [batch_size, 1, 28, 28]
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # output 是模型的原始输出，形状为 [batch_size, 10]
        # 例如 batch_size=128，10个类别（数字0-9）
        # output 的值是每个类别的得分（logits），不是概率

        # output.max(1) 返回两个值：
        # - 最大值（各样本的最高得分）
        # - 最大值对应的索引（预测的类别）

        _, predicted = output.max(1)
        # _  : 最高得分（这里用下划线忽略，不需要）
        # predicted: 预测的类别索引，形状 [128]

        # target.size(0) 返回当前批次的大小（batch_size）
        # 假设 batch_size=128
        total += target.size(0)
        # 拆解这一行：
        # 1. predicted.eq(target)  # 逐元素比较，返回布尔张量
        # 2. .sum()                # 统计 True 的个数
        # 3. .item()               # 将标量张量转为 Python 数字
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            print('Train Accuracy: {:.2f}%'.format(100. * correct / total))
            print('-' * 10)


if __name__ == '__main__':
    batch_size = 128
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, normalize=True)
    print(f"训练集大小：{len(train_loader.dataset)}")

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("layer1 = ", model.layer1.weight.data)
    print("layer2 = ", model.layer2.weight.data)
    print("layer3 = ", model.layer3.weight.data)
    print("layer4 = ", model.layer4.weight.data)
    print("layer5 = ", model.layer5.weight.data)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch)
    print("layer1 = ", model.layer1.weight.data)
    print("layer2 = ", model.layer2.weight.data)
    print("layer3 = ", model.layer3.weight.data)
    print("layer4 = ", model.layer4.weight.data)
    print("layer5 = ", model.layer5.weight.data)
