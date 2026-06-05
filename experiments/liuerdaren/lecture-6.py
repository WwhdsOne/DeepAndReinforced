import torch
import matplotlib.pyplot as plt


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


if __name__ == "__main__":
    model = LogisticRegressionModel()
    criterion = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 生成有逻辑关系的数据
    torch.manual_seed(42)
    x_data = torch.randn(500, 1) * 2
    prob = torch.sigmoid(2 * x_data - 1)  # 真实关系
    y_data = (torch.rand(500, 1) < prob).float()

    # 训练
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        if epoch % 200 == 199:
            # 计算准确率
            y_pred_class = (y_pred > 0.5).float()
            accuracy = (y_pred_class == y_data).float().mean()
            print(
                f"epoch = {epoch}, loss = {loss.item():.4f}, accuracy = {accuracy:.4f}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 正确的可视化：散点图 + 决策边界
    x_sorted, indices = torch.sort(x_data, dim=0)
    y_prob_sorted = model(x_sorted)

    plt.figure(figsize=(10, 6))

    # 分开画两类数据点
    mask_0 = (y_data == 0).squeeze()
    mask_1 = (y_data == 1).squeeze()

    plt.scatter(
        x_data[mask_0].numpy(),
        y_data[mask_0].numpy(),
        c="blue",
        label="Class 0",
        alpha=0.6,
        s=30,
    )
    plt.scatter(
        x_data[mask_1].numpy(),
        y_data[mask_1].numpy(),
        c="red",
        label="Class 1",
        alpha=0.6,
        s=30,
    )

    # 画预测概率曲线
    plt.plot(
        x_sorted.numpy(),
        y_prob_sorted.detach().numpy(),
        "g-",
        label="Predicted probability",
        linewidth=3,
    )

    # 画决策边界 (概率=0.5)
    plt.axhline(
        y=0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary (0.5)"
    )

    plt.xlabel("x")
    plt.ylabel("Probability / Class")
    plt.legend()
    plt.title(
        f"Logistic Regression: w={model.linear.weight.item():.3f}, b={model.linear.bias.item():.3f}"
    )
    plt.grid(True, alpha=0.3)
    plt.show()
