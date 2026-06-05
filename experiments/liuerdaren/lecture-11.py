import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from data_utils import get_mnist_loaders


class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入 通道数：1 卷积核大小：5 填充：2
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(88, 20, 5, padding=2)
        # 策略：快速缩小空间尺寸，减少计算量
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(3168, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        # 向量拍扁
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, 1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, 1)
        self.branch5x5_2 = nn.Conv2d(16, 24, 5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, 1)
        self.branch3x3_2 = nn.Conv2d(16, 24, 3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, 3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [
            branch1x1,
            branch5x5,
            branch3x3,
            branch_pool,
        ]
        return torch.cat(outputs, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)

        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
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
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            print("Train Accuracy: {:.2f}%".format(100.0 * correct / total))
            print("-" * 10)


if __name__ == "__main__":
    batch_size = 128
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, normalize=True)
    print(f"训练集大小：{len(train_loader.dataset)}")

    model = ResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch)
