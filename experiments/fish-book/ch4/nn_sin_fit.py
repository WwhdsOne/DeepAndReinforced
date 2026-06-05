import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["STHeiti"]  # macOS 常用中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# -----------------------------
# 生成数据
# -----------------------------
x_np = np.linspace(0, 2 * np.pi, 1000)  # [0, 2pi]
y_np = np.sin(x_np / 2)

# 转换为 PyTorch tensor
x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)  # [1000,1]
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)  # [1000,1]


# -----------------------------
# 定义神经网络
# -----------------------------
class SinNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 3),
            nn.Tanh(),
            nn.Linear(3, 3),
            nn.Tanh(),
            nn.Linear(3, 1),  # 输出1
        )

    def forward(self, x):
        return self.net(x)


model = SinNN()

# -----------------------------
# 损失函数 & 优化器
# -----------------------------
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.1)

# -----------------------------
# 训练
# -----------------------------
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    # 计算损失
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# -----------------------------
# 绘图对比
# -----------------------------
y_fit = model(x).detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_np, y_np, label="sin(x)", linewidth=2)
plt.plot(x_np, y_fit, label="NN Fit", linewidth=2, linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("神经网络拟合 sin(x)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
