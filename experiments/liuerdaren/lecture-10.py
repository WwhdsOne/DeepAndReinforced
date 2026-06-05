import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from data_utils import get_mnist_loaders


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # conv输出尺寸 = (输入尺寸 - 卷积核大小 + 2×填充) / 步长 + 1
        # pool输出尺寸 = (输入尺寸 - 池化核大小) / 步长 + 1

        # 输入 通道数：1 卷积核大小：5 填充：2

        # 策略：快速缩小空间尺寸，减少计算量
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 28→28
        self.pool1 = nn.MaxPool2d(2, 2)  # 28→14

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 14→14
        self.pool2 = nn.MaxPool2d(2, 2)  # 14→7

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 7→7
        self.pool3 = nn.MaxPool2d(2, 2)  # 7→3

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch)
