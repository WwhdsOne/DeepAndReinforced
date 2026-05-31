from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
from tqdm import tqdm  # 在文件开头导入

from common import dropout

"""
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
"""

sentiment = [0, 1, 2, 3, 4]
labels = [0, 1, 2, 3, 4]


def collate_fn(batch):
    ids_list, labels_list = zip(*batch)
    ids_padded = pad_sequence(ids_list, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(labels_list)
    return ids_padded, labels_tensor


class SentimentDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(
            csv_file,
            sep='\t'
        )

        counter = Counter()
        for idx, sentence in enumerate(self.df["Phrase"]):
            counter.update(sentence.lower().split())

        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
        }

        for word, _ in counter.items():
            self.vocab[word] = len(self.vocab)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["Phrase"]
        label = self.df.iloc[idx]["Sentiment"]

        # 分词
        words = text.lower().split()

        # 转 id
        ids = [
            self.vocab.get(word, self.vocab["<unk>"])
            for word in words
        ]

        ids = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)  # 显式转为 Tensor
        return ids, label


class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = SentimentDataset('data/sentiment-analysis-on-movie-reviews/train.tsv')

    # 假设你已经有了完整的训练数据集对象 train_dataset（包含所有训练样本）
    total_len = len(train_dataset)
    val_ratio = 0.2  # 20% 作为验证集
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_subset, val_subset = random_split(
        train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子，保证可复现
    )

    train_loader = DataLoader(train_subset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(train_dataset.vocab) + 1
    embedding_dim = 64
    hidden_dim = 128
    num_classes = len(sentiment)

    print("总样本数：", total_len)
    print("训练集大小：", train_len)
    print("验证集大小：", val_len)
    print("字典大小：", vocab_size)

    model = SentimentModel(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_time = 20

    for i in range(epoch_time):
        # ----- 训练阶段 -----
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {i + 1}/{epoch_time} [Train]", leave=False)
        for sentence, label in loop:
            sentence, label = sentence.to(device), label.to(device)  # 加这行
            optimizer.zero_grad()
            outputs = model(sentence)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # ----- 验证阶段 -----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {i + 1}/{epoch_time} [Val]", leave=False)
            for sentence, label in val_loop:
                sentence = sentence.to(device)
                label = label.to(device)
                outputs = model(sentence)
                preds = outputs.argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)
                val_loop.set_postfix(acc=correct / total if total > 0 else 0)

        acc = correct / total
        tqdm.write(f"Epoch [{i + 1}/{epoch_time}] Loss={avg_loss:.4f} Acc={acc:.4f}")
