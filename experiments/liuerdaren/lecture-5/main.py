import torch

from ch8.train_kaggle import criterion


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


if __name__ == '__main__':
    x_data = torch.tensor([[1.0], [2.0], [3.0]])
    y_data = torch.tensor([[2.0], [4.0], [6.0]])
    model = LinearModel()
    """
    Mean Squared Error
    中文翻译为：均方误差
    并且计算损失时不对批次中的样本求平均，而是直接求和
    """
    criterion = torch.nn.MSELoss(reduction='sum')
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
