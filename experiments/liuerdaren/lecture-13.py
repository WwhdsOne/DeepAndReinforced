import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence


# ============================
# DataLoader 拼 batch 时调用
# 作用：
# 1. 对不同长度名字进行 padding
# 2. 把 label 转成 Tensor
# ============================
def collate_fn(batch):
    names, labels = zip(*batch)

    names = pad_sequence(
        names,
        batch_first=True,
        padding_value=0
    )

    # CrossEntropyLoss 要求 label 为 LongTensor
    labels = torch.tensor(labels, dtype=torch.long)

    return names, labels


class NameDataset(Dataset):
    def __init__(self, csv_file, char2idx=None, country2idx=None):
        self.df = pd.read_csv(csv_file, header=None)

        self.names = self.df.iloc[:, 0].tolist()
        countries = self.df.iloc[:, 1].tolist()

        # ============================
        # 修改1：
        # 字符索引从 1 开始
        #
        # 0 预留给 padding
        #
        # padding_idx=0 时
        # embedding 不会更新这一行参数
        # ============================
        if char2idx is None:
            unique_chars = sorted(set("".join(self.names)))

            self.char2idx = {
                char: i + 1
                for i, char in enumerate(unique_chars)
            }
        else:
            self.char2idx = char2idx

        # 国家映射
        if country2idx is None:
            unique_countries = sorted(set(countries))

            self.country2idx = {
                country: i
                for i, country in enumerate(unique_countries)
            }
        else:
            self.country2idx = country2idx

        self.countries = [
            self.country2idx[c]
            for c in countries
        ]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        name_ids = [
            self.char2idx[c]
            for c in name
        ]

        return torch.tensor(name_ids), self.countries[idx]


class RNNClassifier(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            output_size
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_size * 2, output_size)   # 因为双向，所以要 *2

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.gru(x)          # out: (batch, seq_len, hidden_size*2)
        last_out = out[:, -1, :]           # (batch, hidden_size*2)
        last_out = self.dropout(last_out)
        return self.fc(last_out)


if __name__ == "__main__":
    train_dataset = NameDataset(
        "data/names_train.csv"
    )

    # ============================
    # 修改2：
    # 测试集必须复用训练集映射
    #
    # 否则：
    # English -> 0
    # Chinese -> 1
    #
    # 和训练阶段可能不一致
    # ============================
    test_dataset = NameDataset(
        "data/names_test.csv",
        train_dataset.char2idx,
        train_dataset.country2idx
    )

    # ============================
    # 修改3：
    # +1 是因为预留了 PAD=0
    # ============================
    vocab_size = len(train_dataset.char2idx) + 1

    output_size = len(train_dataset.country2idx)

    hidden_size = 256
    embedding_dim = 128

    # ============================
    # 新增：
    # DataLoader
    # ============================
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = RNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size
    )

    criterion = nn.CrossEntropyLoss()

    # ============================
    # 新增：
    # Adam 优化器
    # ============================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 10

    for epoch in range(epochs):

        # ============================
        # 训练模式
        # ============================
        model.train()

        total_loss = 0

        for names, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(names)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ============================
        # 测试
        # ============================
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for names, labels in test_loader:
                outputs = model(names)

                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()

                total += labels.size(0)

        acc = correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss={avg_loss:.4f} "
            f"Acc={acc:.4f}"
        )
