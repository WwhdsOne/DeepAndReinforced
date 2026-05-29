import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size,
                 embedding_size, batch_size, num_class, num_layers=1):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.emb = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.view(-1, self.num_class)

    def print_embeddings(self):
        """打印embedding层的向量"""
        print("\n" + "="*50)
        print("Embedding层向量 ({}个词, 每个{}维):".format(self.emb.num_embeddings, self.emb.embedding_dim))
        print("="*50)
        for idx in range(self.emb.num_embeddings):
            vector = self.emb.weight[idx].detach().numpy()
            print(f"词 '{idx2char[idx]}' (ID:{idx}) 的向量: {vector}")
        print("="*50 + "\n")


if __name__ == '__main__':
    num_class = 4
    input_size = 4
    hidden_size = 8
    embedding_size = 10
    num_layers = 2
    batch_size = 1
    seq_len = 5

    idx2char = ['e', 'h', 'l', 'o']
    x_data = [[1, 0, 2, 2, 3]]
    y_data = [3, 1, 2, 3, 2]

    inputs = torch.LongTensor(x_data)
    labels = torch.LongTensor(y_data)

    # 创建模型
    net = Model(input_size=input_size,
                hidden_size=hidden_size,
                embedding_size=embedding_size,
                batch_size=batch_size,
                num_class=num_class,
                num_layers=num_layers)

    # 打印初始化后的embedding向量（随机初始化）
    print("\n【训练前的Embedding向量】")
    net.print_embeddings()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    net.train()
    epoch_times = 10

    for epoch in range(epoch_times):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 每隔几个epoch打印一次预测结果
        predictions = outputs.max(1)[1]
        print('Predict = ', ''.join([idx2char[idx] for idx in predictions]))
        print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, epoch_times, loss.item()))

        # 每5个epoch打印一次embedding向量变化
        if (epoch + 1) % 5 == 0:
            print(f"\n【训练 {epoch+1} 个epoch后的Embedding向量】")
            net.print_embeddings()

    # 最终打印训练好的embedding向量
    print("\n【最终训练好的Embedding向量】")
    net.print_embeddings()