import random
from functools import partial
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class Dictionary:
    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = []
        for token in ["<s>", "<pad>", "</s>", "<unk>"]:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.tok2idx:
            self.tok2idx[token] = len(self.idx2tok)
            self.idx2tok.append(token)

    def __len__(self):
        return len(self.idx2tok)

    def pad(self):
        return self.tok2idx["<pad>"]

    def bos(self):
        return self.tok2idx["<s>"]

    def eos(self):
        return self.tok2idx["</s>"]

    def unk(self):
        return self.tok2idx["<unk>"]

    def encode(self, tokens):
        return [self.tok2idx.get(token, self.unk()) for token in tokens]

    def string(self, tensor):
        tokens = [
            self.idx2tok[int(index)]
            for index in tensor
            if int(index) not in {self.pad(), self.bos(), self.eos()}
        ]
        return " ".join(tokens).replace(" ", "").replace("\u2581", " ").strip()


def build_dictionary(spm_model_path):
    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    dictionary = Dictionary()
    for index in range(sp.get_piece_size()):
        dictionary.add_token(sp.id_to_piece(index))
    return dictionary


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file,
        tgt_file,
        src_dict,
        tgt_dict,
        max_src_len=160,
        max_tgt_len=160,
        max_samples=None,
    ):
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_data = []
        self.tgt_data = []
        self.src_lengths = []
        self.tgt_lengths = []
        self.filtered_count = 0

        with Path(src_file).open() as src_f, Path(tgt_file).open() as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()
                src_len = len(src_tokens) + 1
                tgt_len = len(tgt_tokens) + 1
                if src_len > max_src_len or tgt_len > max_tgt_len:
                    self.filtered_count += 1
                    continue
                self.src_data.append(src_dict.encode(src_tokens))
                self.tgt_data.append(tgt_dict.encode(tgt_tokens))
                self.src_lengths.append(src_len)
                self.tgt_lengths.append(tgt_len)

        if max_samples is not None:
            self.src_data = self.src_data[:max_samples]
            self.tgt_data = self.tgt_data[:max_samples]
            self.src_lengths = self.src_lengths[:max_samples]
            self.tgt_lengths = self.tgt_lengths[:max_samples]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, index):
        return {
            "source": torch.tensor(
                self.src_data[index] + [self.src_dict.eos()], dtype=torch.long
            ),
            "target": torch.tensor(
                self.tgt_data[index] + [self.tgt_dict.eos()], dtype=torch.long
            ),
        }


class TokenBatchSampler(Sampler):
    def __init__(self, dataset, max_tokens, shuffle=False, seed=73, bucket_size=1024):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size
        self.lengths = [
            max(src_len, tgt_len)
            for src_len, tgt_len in zip(dataset.src_lengths, dataset.tgt_lengths)
        ]

    def __iter__(self):
        rng = random.Random(self.seed)
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index], reverse=True)
            batch = []
            batch_max_len = 0
            for index in bucket:
                next_max_len = max(batch_max_len, self.lengths[index])
                if batch and next_max_len * (len(batch) + 1) > self.max_tokens:
                    batches.append(batch)
                    batch = []
                    batch_max_len = 0
                batch.append(index)
                batch_max_len = max(batch_max_len, self.lengths[index])
            if batch:
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum(1 for _ in self.__iter__())


def collate_translation_batch(samples, pad_idx, eos_idx):
    batch_size = len(samples)
    src_len = max(len(sample["source"]) for sample in samples)
    tgt_len = max(len(sample["target"]) for sample in samples)
    src_tokens = torch.full((batch_size, src_len), pad_idx, dtype=torch.long)
    target = torch.full((batch_size, tgt_len), pad_idx, dtype=torch.long)
    src_lengths = torch.zeros(batch_size, dtype=torch.long)

    for index, sample in enumerate(samples):
        src_tokens[index, : len(sample["source"])] = sample["source"]
        target[index, : len(sample["target"])] = sample["target"]
        src_lengths[index] = len(sample["source"])

    prev_output_tokens = torch.full((batch_size, tgt_len), pad_idx, dtype=torch.long)
    prev_output_tokens[:, 0] = eos_idx
    prev_output_tokens[:, 1:] = target[:, :-1]

    return {
        "nsentences": batch_size,
        "ntokens": int((target != pad_idx).sum().item()),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": prev_output_tokens,
        },
        "target": target,
    }


def build_dataloader(dataset, dictionary, max_tokens, epoch=1, shuffle=False, num_workers=0):
    sampler = TokenBatchSampler(
        dataset,
        max_tokens=max_tokens,
        shuffle=shuffle,
        seed=73 + epoch,
    )
    collate_fn = partial(
        collate_translation_batch,
        pad_idx=dictionary.pad(),
        eos_idx=dictionary.eos(),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def load_parallel_dataset(data_dir, split, dictionary, max_src_len, max_tgt_len):
    data_dir = Path(data_dir)
    return TranslationDataset(
        data_dir / f"{split}.en",
        data_dir / f"{split}.zh",
        dictionary,
        dictionary,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
