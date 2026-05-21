"""TwoLayerNet: 基于 PyTorch 的手写数字识别（对应 Ch6 的 NumPy 实现）。"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class MultiLayerNet(nn.Module):
    """双层全连接网络，含 BatchNorm、Dropout，对应 NumPy 版的 MultiLayerNet。"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.1):
        super().__init__()
        # 第一层：Affine -> BatchNorm -> ReLU -> Dropout
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第二层：Affine -> BatchNorm
        self.affine2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)

        # 应用 Kaiming / Xavier 初始化（与 NumPy 版保持一致）
        self._init_weights(input_size, hidden_size, output_size)

    def _init_weights(self, input_size, hidden_size, output_size):
        """Kaiming 初始化（ReLU 适用）和 Xavier 初始化。"""
        # W1: Kaiming 初始化（He initialization）
        nn.init.kaiming_normal_(self.affine1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.affine1.bias)

        # W2: Xavier 初始化
        nn.init.xavier_normal_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。训练/推理模式由 self.training 自动控制 Dropout 和 BatchNorm。"""
        # 第一层：Affine -> BatchNorm -> Dropout -> ReLU
        x = self.bn1(self.affine1(x))
        x = self.dropout1(x)  # Dropout 根据 self.training 自动决定是否丢弃
        x = F.relu(x)

        # 第二层：Affine -> BatchNorm（输出层，无 ReLU）
        x = self.bn2(self.affine2(x))

        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理时使用（Dropout 关闭，BatchNorm 用滑动均值/方差）。"""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def accuracy(self, x: torch.Tensor, t: torch.Tensor) -> float:
        """计算准确率。"""
        self.eval()
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        labels = torch.argmax(t, dim=1) if t.ndim > 1 else t
        return (preds == labels).float().mean().item()


def train_and_test(
    input_size: int = 784,
    hidden_size: int = 256,
    output_size: int = 10,
    iters_num: int = 6000,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    lambda_reg: float = 1e-4,
    data_dir: str = None,
):
    """训练并测试 MultiLayerNet，对应 NumPy 版的 trainAndTest()。"""
    # 默认数据目录：与 train_nn.py 保持一致
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "../data")
    """训练并测试 MultiLayerNet，对应 NumPy 版的 trainAndTest()。"""
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 MNIST 数据（使用 torchvision，归一化到 [0, 1]）
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # 将数据集转换为展平的 numpy 数组，再转为 tensor（与 NumPy 版预处理方式保持一致）
    x_train = train_dataset.data.float().view(-1, 784) / 255.0
    t_train = train_dataset.targets
    x_test = test_dataset.data.float().view(-1, 784) / 255.0
    t_test = test_dataset.targets

    # 转为 one-hot（与 NumPy 版保持一致，方便直接对比）
    def to_one_hot(labels, num_classes):
        return torch.eye(num_classes)[labels]

    t_train_onehot = to_one_hot(t_train, output_size)
    t_test_onehot = to_one_hot(t_test, output_size)

    # 构建网络
    network = MultiLayerNet(input_size, hidden_size, output_size).to(device)

    # Adam 优化器（PyTorch 的 Adam 内置 L2 正则化通过 weight_decay 参数）
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=lambda_reg)

    train_size = len(x_train)
    iter_per_epoch = max(train_size // batch_size, 1)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 学习率调度：每个 epoch 衰减为原来的 0.9（与 NumPy 版一致）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iter_per_epoch, gamma=0.9)

    network.train()  # 训练模式：Dropout 开启，BatchNorm 用 batch 统计量
    for i in range(iters_num):
        # 随机采样 batch
        batch_mask = torch.randint(0, train_size, (batch_size,))
        x_batch = x_train[batch_mask].to(device)
        t_batch = t_train_onehot[batch_mask].to(device)

        # 前向传播 + 计算损失（network.train() 已设置，Dropout/BatchNorm 自动进入训练模式）
        logits = network(x_batch)
        loss = F.cross_entropy(logits, torch.argmax(t_batch, dim=1))

        # L2 正则化已通过 weight_decay 在优化器中实现，无需手动添加

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss_list.append(loss.item())

        # 每个 epoch 评估一次
        if i % iter_per_epoch == 0:
            x_train_dev = x_train.to(device)
            t_train_dev = t_train_onehot.to(device)
            x_test_dev = x_test.to(device)
            t_test_dev = t_test_onehot.to(device)

            train_acc = network.accuracy(x_train_dev, t_train_dev)
            test_acc = network.accuracy(x_test_dev, t_test_dev)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"iter {i:5d} | loss {loss.item():.4f} | "
                f"train acc {train_acc:.4f} | test acc {test_acc:.4f} | "
                f"lr {current_lr:.6f}"
            )

    print(f"\n最终训练准确率: {train_acc_list[-1]:.4f}")
    print(f"最终测试准确率: {test_acc_list[-1]:.4f}")

    return train_acc_list[-1], test_acc_list[-1]


if __name__ == "__main__":
    avg_train_acc, avg_test_acc = 0.0, 0.0
    loop_count = 10
    for i in range(loop_count):
        train_acc, test_acc = train_and_test()
        avg_train_acc += train_acc
        avg_test_acc += test_acc
    print(f"平均训练准确率: {avg_train_acc / loop_count:.4f}")
    print(f"平均测试准确率: {avg_test_acc / loop_count:.4f}")
