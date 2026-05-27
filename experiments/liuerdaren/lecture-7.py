import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体为黑体
plt.rcParams['font.family'] = 'STHeiti'
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 1)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.linear3(x)  # 最后一层无激活，输出 logits
        return x

if __name__ == '__main__':
    model = Model()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    xy = np.loadtxt('../data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
    # 假设 xy 是这样的数据（特征列 + 标签列）：
    # xy = [[特征1, 特征2, ..., 特征n, 标签],
    #       [特征1, 特征2, ..., 特征n, 标签],
    #       ...]

    x_data = torch.from_numpy(xy[:,:-1])    # 取所有行，除了最后一列
    y_data = torch.from_numpy(xy[:, [-1]])  # 取所有行，只取最后一列

    # 训练
    for epoch in range(2000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        if epoch % 400 == 199:
            # 计算准确率
            y_pred_class = (y_pred > 0).float()
            accuracy = (y_pred_class == y_data).float().mean()
            print(f"epoch = {epoch + 1}, loss = {loss.item():.4f}, accuracy = {accuracy:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 训练结束后计算最终预测和准确率
    with torch.no_grad():
        y_pred = model(x_data)
        y_pred_class = (y_pred > 0).float()
    final_accuracy = (y_pred_class == y_data).float().mean().item()

    correct = (y_pred_class == y_data).squeeze()
    incorrect = ~correct

    # 可视化：左右对比图 — 真实标签 vs 模型预测
    plt.figure(figsize=(14, 6))

    # 左图：真实标签
    plt.subplot(1, 2, 1)
    mask_0 = (y_data == 0).squeeze()
    mask_1 = (y_data == 1).squeeze()
    plt.scatter(x_data[mask_0, 0].numpy(), x_data[mask_0, 1].numpy(),
                c='blue', label='Class 0', alpha=0.7, s=30)
    plt.scatter(x_data[mask_1, 0].numpy(), x_data[mask_1, 1].numpy(),
                c='red', label='Class 1', alpha=0.7, s=30)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('Ground Truth（真实标签）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：模型预测结果
    plt.subplot(1, 2, 2)
    pred_mask_0 = (y_pred_class == 0).squeeze()
    pred_mask_1 = (y_pred_class == 1).squeeze()
    plt.scatter(x_data[pred_mask_0, 0].numpy(), x_data[pred_mask_0, 1].numpy(),
                c='blue', label='Pred Class 0', alpha=0.7, s=30)
    plt.scatter(x_data[pred_mask_1, 0].numpy(), x_data[pred_mask_1, 1].numpy(),
                c='red', label='Pred Class 1', alpha=0.7, s=30)
    # 用黑色空心圆标出预测错误的点
    plt.scatter(x_data[incorrect, 0].numpy(), x_data[incorrect, 1].numpy(),
                facecolors='none', edgecolors='black', linewidths=2, s=80, label='Misclassified')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title(f'Model Prediction（模型预测）— Acc: {final_accuracy:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Diabetes Classification: Ground Truth vs Model Prediction', fontsize=14)
    plt.tight_layout()
    plt.show()