import torch
import matplotlib.pyplot as plt


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


if __name__ == '__main__':
    x_data = torch.rand(500, 1)
    y_data = 2 * x_data + 0.1 * torch.rand(500, 1)

    model = LinearModel()
    """
    Mean Squared Error
    均方误差
    """
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(500):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        if epoch % 100 == 99:
            print("epoch = ", epoch, "loss = ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('w =', model.linear.weight.item())
    print('b =', model.linear.bias.item())

    # 关键：对 x 排序后再画预测线
    x_sorted, indices = torch.sort(x_data, dim=0)
    y_pred_sorted = model(x_sorted)

    plt.figure(figsize=(8, 6))
    plt.plot(x_data.numpy(), y_data.numpy(), 'o', label='Data', alpha=0.5, markersize=3)
    plt.plot(x_sorted.numpy(), y_pred_sorted.detach().numpy(), 'r-', label='Fitted line', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Linear Regression: w={model.linear.weight.item():.3f}, b={model.linear.bias.item():.3f}')
    plt.show()