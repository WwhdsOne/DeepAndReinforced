import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

"""
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
"""

sentiment = [0, 1, 2, 3, 4]
labels = [0, 1, 2, 3, 4]


class SentimentDataset(Dataset):
    def __init__(self, csv_file, vocab=None):
        self.df = pd.read_csv(
            csv_file,
            sep='\t'
        )

        if vocab is None:
            print(csv_file)
            counter = Counter()
            for idx, sentence in enumerate(self.df["Phrase"]):
                if not isinstance(sentence, str):
                    print(
                        f"row={idx}, value={sentence}, "
                        f"type={type(sentence)}"
                    )
                    continue
                counter.update(sentence.lower().split())

            self.vocab = {
                "<pad>": 0,
                "<unk>": 1,
            }

            for word, _ in counter.items():
                self.vocab[word] = len(self.vocab)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        label = self.df.iloc[idx]["label"]

        # 分词
        words = text.lower().split()

        # 转 id
        ids = [
            self.vocab.get(word, self.vocab["<unk>"])
            for word in words
        ]

        ids = torch.tensor(ids, dtype=torch.long)

        return ids, label

if __name__ == "__main__":
    train_dataset = SentimentDataset('data/sentiment-analysis-on-movie-reviews/train.tsv')

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
