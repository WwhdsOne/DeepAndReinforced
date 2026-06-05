import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )

    def forward(self, input, hidden):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


if __name__ == "__main__":
    input_size = 4
    hidden_size = 4
    batch_size = 1
    seq_len = 5

    idx2char = ["e", "h", "l", "o"]
    x_data = [1, 0, 2, 2, 3]
    y_data = [3, 1, 2, 3, 2]

    print("x = ", "".join([idx2char[x] for x in x_data]))
    print("y = ", "".join([idx2char[y] for y in y_data]))

    one_hot_lookup = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    x_one_hot = [one_hot_lookup[x] for x in x_data]
    print("x_one_shot = ", x_one_hot)
    inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
    labels = torch.LongTensor(y_data)
    # print("inputs = ", inputs)
    # print("labels = ", labels)

    net = Model(input_size, hidden_size, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    net.train()
    epoch_times = 10
    for epoch in range(epoch_times):
        optimizer.zero_grad()  # 清空梯度
        hidden = net.init_hidden()  # 初始化隐藏状态
        total_loss = 0

        print("Predict = ", end="")

        # 遍历每个时间步
        hidden = net(inputs, hidden)  # 前向传播
        loss = criterion(hidden, labels)  # 当前步的损失

        # 获取预测的字符索引
        _, idxs = hidden.max(dim=1)
        print("".join([idx2char[idx] for idx in idxs]), end="")

        # 在所有时间步完成后，反向传播一次
        loss.backward()  # 这里用 total_loss 而不是 loss
        optimizer.step()

        print(", Epoch [%d/%d] loss=%.4f" % (epoch + 1, epoch_times, loss.item()))
