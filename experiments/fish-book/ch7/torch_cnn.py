"""SimpleConvNet: 基于 PyTorch 的卷积神经网络（对应 Ch7 的 NumPy 实现）。"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class TorchSimpleConvNet(nn.Module):
    """简单卷积神经网络，对应 NumPy 版的 SimpleConvNet。

    网络结构：
    Conv1 → ReLU → Pool1 → Affine1 → ReLU2 → Affine2 → Softmax
    """

    def __init__(self, input_dim=(1, 28, 28), filter_num=30, filter_size=5,
                 hidden_size=50, output_size=10, stride=1, pad=0):
        super().__init__()

        # 计算卷积输出尺寸
        input_h, input_w = input_dim[1], input_dim[2]
        conv_output_h = (input_h + 2 * pad - filter_size) // stride + 1
        conv_output_w = (input_w + 2 * pad - filter_size) // stride + 1

        # 池化后尺寸（池化窗口 2x2，步幅 2）
        pool_output_h = conv_output_h // 2
        pool_output_w = conv_output_w // 2

        # 全连接层的输入尺寸
        self.pool_output_size = filter_num * pool_output_h * pool_output_w

        # 卷积层
        self.conv1 = nn.Conv2d(
            in_channels=input_dim[0],
            out_channels=filter_num,
            kernel_size=filter_size,
            stride=stride,
            padding=pad
        )

        # 全连接层
        self.fc1 = nn.Linear(self.pool_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 初始化权重（与 NumPy 版保持一致）
        self._init_weights(filter_num, filter_size, input_dim, hidden_size, output_size)

    def _init_weights(self, filter_num, filter_size, input_dim, hidden_size, output_size):
        """初始化权重（对应 NumPy 版的初始化方式）。"""
        # 卷积层权重：Xavier 初始化
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        # 全连接层权重：Xavier 初始化
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        # Conv1 → ReLU → Pool1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # 展平：(N, C, H, W) -> (N, C*H*W)
        x = x.view(x.size(0), -1)

        # Affine1 → ReLU2 → Affine2
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 logits（Softmax 在损失函数中）

        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理时使用（返回 softmax 概率）。"""
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
    input_dim=(1, 28, 28),
    filter_num=30,
    filter_size=5,
    hidden_size=50,
    output_size=10,
    iters_num=10,
    batch_size=128,
    learning_rate=0.001,
    data_dir=None,
):
    """训练并测试 TorchSimpleConvNet，对应 NumPy 版的 trainAndTest()。"""
    # 默认数据目录
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "../data")

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 MNIST 数据
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # 准备数据（与 NumPy 版保持一致）
    x_train = train_dataset.data.float().view(-1, 1, 28, 28) / 255.0
    t_train = train_dataset.targets
    x_test = test_dataset.data.float().view(-1, 1, 28, 28) / 255.0
    t_test = test_dataset.targets

    # 转为 one-hot
    def to_one_hot(labels, num_classes):
        return torch.eye(num_classes)[labels]

    t_train_onehot = to_one_hot(t_train, output_size)
    t_test_onehot = to_one_hot(t_test, output_size)

    # 构建网络
    network = TorchSimpleConvNet(
        input_dim=input_dim,
        filter_num=filter_num,
        filter_size=filter_size,
        hidden_size=hidden_size,
        output_size=output_size
    ).to(device)

    # Adam 优化器
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    train_size = len(x_train)
    iter_per_epoch = max(train_size // batch_size, 1)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 学习率调度：每个 epoch 衰减为原来的 0.9
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iter_per_epoch, gamma=0.9)

    # 训练循环
    network.train()
    for i in range(iters_num):
        # 随机采样 batch
        batch_mask = torch.randint(0, train_size, (batch_size,))
        x_batch = x_train[batch_mask].to(device)
        t_batch = t_train_onehot[batch_mask].to(device)

        # 前向传播 + 计算损失
        logits = network(x_batch)
        loss = F.cross_entropy(logits, torch.argmax(t_batch, dim=1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss_list.append(loss.item())

        # 每 5 个 epoch 评估一次
        eval_interval = iter_per_epoch * 5
        if i % eval_interval == 0 or i == iters_num - 1:
            network.eval()
            with torch.no_grad():
                # 分批计算准确率（避免内存爆炸）
                train_acc = _batch_accuracy(network, x_train.to(device), t_train_onehot.to(device), batch_size)
                test_acc = _batch_accuracy(network, x_test.to(device), t_test_onehot.to(device), batch_size)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"iter {i:5d} | loss {loss.item():.4f} | "
                f"train acc {train_acc:.4f} | test acc {test_acc:.4f} | "
                f"lr {current_lr:.6f}"
            )
            network.train()

    # 最终评估
    final_train_acc = _batch_accuracy(network, x_train.to(device), t_train_onehot.to(device), batch_size)
    final_test_acc = _batch_accuracy(network, x_test.to(device), t_test_onehot.to(device), batch_size)
    print(f"\n最终训练准确率: {final_train_acc:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")

    return final_train_acc, final_test_acc


@torch.no_grad()
def _batch_accuracy(network, x, t, batch_size=2000):
    """分批计算准确率（避免内存爆炸）。"""
    total_correct = 0
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i + batch_size]
        t_batch = t[i:i + batch_size]
        logits = network(x_batch)
        preds = torch.argmax(logits, dim=1)
        labels = torch.argmax(t_batch, dim=1) if t_batch.ndim > 1 else t_batch
        total_correct += (preds == labels).sum().item()
    return total_correct / float(x.size(0))


if __name__ == "__main__":
    avg_train_acc, avg_test_acc = 0.0, 0.0
    loop_count = 1
    for i in range(loop_count):
        train_acc, test_acc = train_and_test()
        avg_train_acc += train_acc
        avg_test_acc += test_acc
    print(f"平均训练准确率: {avg_train_acc / loop_count:.4f}")
    print(f"平均测试准确率: {avg_test_acc / loop_count:.4f}")
