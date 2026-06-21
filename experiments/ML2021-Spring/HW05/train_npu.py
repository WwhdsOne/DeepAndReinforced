#!/usr/bin/env python3
"""
HW05 Seq2Seq 机器翻译 — 多设备独立训练脚本

支持设备：CUDA / NPU (升腾910, BF16) / MPS / CPU
自动下载和预处理数据集（已存在则跳过）。

用法：
    python train_npu.py                              # 默认参数
    python train_npu.py --max-epoch 10 --beam 3      # 自定义参数
    python train_npu.py --predict                    # 仅做预测
    python train_npu.py --no-bf16                    # NPU 上禁用 BF16
"""

import sys
import os
import re
import math
import random
import shutil
import logging
import argparse
import subprocess
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    from tqdm.auto import tqdm

import sentencepiece as spm
import sacrebleu

# ============================================================
# 数据下载与预处理
# ============================================================


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


def clean_s(s, lang):
    if lang == "en":
        s = re.sub(r"\([^()]*\)", "", s)
        s = s.replace("-", "")
        s = re.sub('([.,;!?()"])', r" \1 ", s)
    elif lang == "zh":
        s = strQ2B(s)
        s = re.sub(r"\([^()]*\)", "", s)
        s = s.replace(" ", "").replace("—", "")
        s = s.replace('"', '"').replace('"', '"').replace("_", "")
        s = re.sub('([。,;!?()"~「」])', r" \1 ", s)
    return " ".join(s.strip().split())


def len_s(s, lang):
    return len(s) if lang == "zh" else len(s.split())


def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f"{prefix}.clean.{l1}").exists() and Path(f"{prefix}.clean.{l2}").exists():
        print(f"clean files exist, skipping.")
        return
    with (
        open(f"{prefix}.{l1}") as l1_in,
        open(f"{prefix}.{l2}") as l2_in,
        open(f"{prefix}.clean.{l1}", "w") as l1_out,
        open(f"{prefix}.clean.{l2}", "w") as l2_out,
    ):
        for s1 in l1_in:
            s1, s2 = s1.strip(), l2_in.readline().strip()
            s1, s2 = clean_s(s1, l1), clean_s(s2, l2)
            s1_len, s2_len = len_s(s1, l1), len_s(s2, l2)
            if min_len > 0 and (s1_len < min_len or s2_len < min_len):
                continue
            if max_len > 0 and (s1_len > max_len or s2_len > max_len):
                continue
            if ratio > 0 and (s1_len / s2_len > ratio or s2_len / s1_len > ratio):
                continue
            print(s1, file=l1_out)
            print(s2, file=l2_out)


def prepare_data(
    data_dir="./DATA/rawdata",
    dataset_name="ted2020",
    vocab_size=8000,
    valid_ratio=0.01,
    seed=73,
):
    prefix = Path(data_dir).absolute() / dataset_name
    prefix.mkdir(parents=True, exist_ok=True)

    data_prefix = f"{prefix}/train_dev.raw"
    test_prefix = f"{prefix}/test.raw"
    spm_prefix = prefix / f"spm{vocab_size}"

    # 检查是否已全部完成
    if (
        Path(f"{data_prefix}.clean.en").exists()
        and Path(f"{test_prefix}.clean.en").exists()
        and Path(f"{prefix}/train.en").exists()
        and Path(f"{spm_prefix}.model").exists()
    ):
        print("数据已预处理完成，跳过。")
        return str(spm_prefix) + ".model"

    # 1. 下载
    urls = (
        "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.data.tgz",
        "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.test.tgz",
    )
    file_names = ("ted2020.tgz", "test.tgz")

    train_exists = (prefix / "train_dev.raw.en").exists() and (
        prefix / "train_dev.raw.zh"
    ).exists()
    test_exists = (prefix / "test.raw.en").exists() and (
        prefix / "test.raw.zh"
    ).exists()

    if not (train_exists and test_exists):
        print("下载数据集...")
        for u, f in zip(urls, file_names):
            p = prefix / f
            if not p.exists():
                subprocess.run(["wget", u, "-O", str(p)], check=True)
            if p.suffix == ".tgz":
                subprocess.run(["tar", "-xvf", str(p), "-C", str(prefix)], check=True)
        if (prefix / "raw.en").exists():
            (prefix / "raw.en").rename(prefix / "train_dev.raw.en")
        if (prefix / "raw.zh").exists():
            (prefix / "raw.zh").rename(prefix / "train_dev.raw.zh")
        if (prefix / "test.en").exists():
            (prefix / "test.en").rename(prefix / "test.raw.en")
        if (prefix / "test.zh").exists():
            (prefix / "test.zh").rename(prefix / "test.raw.zh")
    else:
        print("数据文件已存在，跳过下载。")

    # 2. 清洗
    print("清洗数据...")
    clean_corpus(data_prefix, "en", "zh")
    clean_corpus(test_prefix, "en", "zh", ratio=-1, min_len=-1, max_len=-1)

    # 3. 切分 train/valid
    if not (
        Path(f"{prefix}/train.clean.en").exists()
        and Path(f"{prefix}/valid.clean.en").exists()
    ):
        print(f"切分 train/valid (valid_ratio={valid_ratio})...")
        line_num = sum(1 for _ in open(f"{data_prefix}.clean.en"))
        labels = list(range(line_num))
        random.seed(seed)
        random.shuffle(labels)
        train_ratio = 1 - valid_ratio
        for lang in ["en", "zh"]:
            train_f = open(prefix / f"train.clean.{lang}", "w")
            valid_f = open(prefix / f"valid.clean.{lang}", "w")
            for count, line in enumerate(open(f"{data_prefix}.clean.{lang}")):
                if labels[count] / line_num < train_ratio:
                    train_f.write(line)
                else:
                    valid_f.write(line)
            train_f.close()
            valid_f.close()
    else:
        print("train/valid 已切分，跳过。")

    # 4. Sentencepiece 训练
    if not Path(f"{spm_prefix}.model").exists():
        print(f"训练 sentencepiece (vocab_size={vocab_size})...")
        spm.SentencePieceTrainer.train(
            input=",".join(
                [
                    f"{prefix}/train.clean.en",
                    f"{prefix}/valid.clean.en",
                    f"{prefix}/train.clean.zh",
                    f"{prefix}/valid.clean.zh",
                ]
            ),
            model_prefix=str(spm_prefix),
            vocab_size=vocab_size,
            character_coverage=1,
            model_type="unigram",
            input_sentence_size=1e6,
            shuffle_input_sentence=True,
            normalization_rule_name="nmt_nfkc_cf",
        )
    else:
        print("sentencepiece model 已存在，跳过。")

    # 5. Sentencepiece 编码
    sp = spm.SentencePieceProcessor(model_file=str(spm_prefix) + ".model")
    in_tag = {"train": "train.clean", "valid": "valid.clean", "test": "test.raw.clean"}
    for split in ["train", "valid", "test"]:
        for lang in ["en", "zh"]:
            out_path = prefix / f"{split}.{lang}"
            if out_path.exists():
                continue
            print(f"编码 {split}.{lang}...")
            with (
                open(out_path, "w") as out_f,
                open(prefix / f"{in_tag[split]}.{lang}") as in_f,
            ):
                for line in in_f:
                    line = line.strip()
                    tok = sp.encode(line, out_type=str)
                    print(" ".join(tok), file=out_f)

    print("数据预处理完成。")
    return str(spm_prefix) + ".model"


# ============================================================
# 设备检测
# ============================================================


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu")
        print(f"Using NPU: {torch.npu.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ============================================================
# 随机种子
# ============================================================


def set_seed(seed=73):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Dictionary
# ============================================================


class Dictionary:
    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = []
        for tok in ["<s>", "<pad>", "</s>", "<unk>"]:
            self.add_token(tok)

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
        return [self.tok2idx.get(t, self.unk()) for t in tokens]

    def decode(self, indices):
        return [self.idx2tok[i] for i in indices]

    def string(self, tensor, post_process="sentencepiece"):
        tokens = [
            self.idx2tok[i]
            for i in tensor
            if i not in {self.pad(), self.bos(), self.eos()}
        ]
        s = " ".join(tokens)
        if post_process == "sentencepiece":
            s = s.replace(" ", "").replace("\u2581", " ").strip()
        return s


def build_dictionary(spm_model_path):
    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    d = Dictionary()
    for i in range(sp.get_piece_size()):
        d.add_token(sp.id_to_piece(i))
    return d


# ============================================================
# Dataset & Task
# ============================================================


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_dict, tgt_dict, max_samples=None):
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_data = []
        self.tgt_data = []

        with open(src_file) as f:
            for line in f:
                tokens = line.strip().split()
                self.src_data.append(src_dict.encode(tokens))

        with open(tgt_file) as f:
            for line in f:
                tokens = line.strip().split()
                self.tgt_data.append(tgt_dict.encode(tokens))

        if max_samples is not None and max_samples < len(self.src_data):
            self.src_data = self.src_data[:max_samples]
            self.tgt_data = self.tgt_data[:max_samples]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return {
            "source": torch.tensor(
                self.src_data[idx] + [self.src_dict.eos()], dtype=torch.long
            ),
            "target": torch.tensor(
                self.tgt_data[idx] + [self.tgt_dict.eos()], dtype=torch.long
            ),
        }


class TranslationTask:
    def __init__(self, data_dir, src_lang, tgt_lang, src_dict, tgt_dict):
        self.data_dir = Path(data_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.datasets = {}

    def load_dataset(self, split, max_samples=None):
        src_file = self.data_dir / f"{split}.{self.src_lang}"
        tgt_file = self.data_dir / f"{split}.{self.tgt_lang}"
        self.datasets[split] = TranslationDataset(
            src_file,
            tgt_file,
            self.src_dict,
            self.tgt_dict,
            max_samples=max_samples,
        )

    def dataset(self, split):
        return self.datasets[split]


# ============================================================
# Collate & DataLoader
# ============================================================


def collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    samples = sorted(samples, key=lambda x: len(x["source"]), reverse=True)
    batch_size = len(samples)
    src_len = max(len(s["source"]) for s in samples)
    tgt_len = max(len(s["target"]) for s in samples)

    src_tokens = torch.full((batch_size, src_len), pad_idx, dtype=torch.long)
    tgt_tokens = torch.full((batch_size, tgt_len), pad_idx, dtype=torch.long)
    src_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, s in enumerate(samples):
        src_tokens[i, : len(s["source"])] = s["source"]
        tgt_tokens[i, : len(s["target"])] = s["target"]
        src_lengths[i] = len(s["source"])

    prev_output_tokens = torch.full((batch_size, tgt_len), pad_idx, dtype=torch.long)
    prev_output_tokens[:, 0] = eos_idx
    prev_output_tokens[:, 1:] = tgt_tokens[:, :-1]

    ntokens = (tgt_tokens != pad_idx).sum().item()

    return {
        "id": list(range(batch_size)),
        "nsentences": batch_size,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": prev_output_tokens,
        },
        "target": tgt_tokens,
    }


def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=0):
    dataset = task.dataset(split)
    avg_len = np.mean(
        [len(dataset[i]["source"]) for i in range(min(100, len(dataset)))]
    )
    batch_size = max(1, int(max_tokens / avg_len))

    custom_collate = partial(
        collate_fn, pad_idx=task.src_dict.pad(), eos_idx=task.src_dict.eos()
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=custom_collate,
        num_workers=num_workers,
    )


# ============================================================
# Model: Transformer
# ============================================================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.embed_tokens = embed_tokens
        self.pos_encoder = PositionalEncoding(args.encoder_embed_dim, args.dropout)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=args.encoder_embed_dim,
                nhead=args.encoder_attention_heads,
                dim_feedforward=args.encoder_ffn_embed_dim,
                dropout=args.dropout,
                batch_first=True,
            )
            for _ in range(args.encoder_layers)
        ])

    def forward(self, src_tokens, src_lengths=None, **unused):
        x = self.embed_tokens(src_tokens)
        x = self.pos_encoder(x)
        src_key_padding_mask = src_tokens.eq(self.padding_idx)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return (x, None, encoder_padding_mask)


class TransformerDecoder(nn.Module):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.embed_dim = args.decoder_embed_dim
        self.embed_tokens = embed_tokens
        self.pos_encoder = PositionalEncoding(args.decoder_embed_dim, args.dropout)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=args.decoder_embed_dim,
                nhead=args.decoder_attention_heads,
                dim_feedforward=args.decoder_ffn_embed_dim,
                dropout=args.dropout,
                batch_first=True,
            )
            for _ in range(args.decoder_layers)
        ])
        self.output_projection = nn.Linear(
            args.decoder_embed_dim, len(dictionary), bias=False
        )
        if args.share_decoder_input_output_embed:
            self.output_projection.weight = embed_tokens.weight

    def forward(
        self, prev_output_tokens, encoder_out, incremental_state=None, **unused
    ):
        encoder_outputs = encoder_out[0]
        tgt = self.embed_tokens(prev_output_tokens)
        tgt = self.pos_encoder(tgt)

        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device
        )
        memory_key_padding_mask = encoder_out[2].t()

        x = tgt
        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_outputs,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        x = self.output_projection(x)
        return x, None


class Seq2Seq(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens=True
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra


def build_model(args, task):
    src_dict, tgt_dict = task.src_dict, task.tgt_dict
    encoder_embed_tokens = nn.Embedding(
        len(src_dict), args.encoder_embed_dim, src_dict.pad()
    )
    decoder_embed_tokens = nn.Embedding(
        len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad()
    )

    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
    model = Seq2Seq(args, encoder, decoder)

    def init_params(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

    model.apply(init_params)
    if args.share_decoder_input_output_embed:
        model.decoder.output_projection.weight = model.decoder.embed_tokens.weight
    return model


# ============================================================
# Optimizer & Criterion
# ============================================================


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def multiply_grads(self, c):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return (
            0
            if not step
            else self.factor
            * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
            )
        )


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss


# ============================================================
# Beam Search
# ============================================================


def strip_pad(tensor, pad_idx):
    return tensor[tensor != pad_idx]


def beam_search_generate(
    model, sample, beam_size=5, max_len_a=1.2, max_len_b=10, device=None
):
    model.eval()
    src_tokens = sample["net_input"]["src_tokens"]
    src_lengths = sample["net_input"]["src_lengths"]
    batch_size = src_tokens.size(0)

    encoder_out = model.encoder(src_tokens, src_lengths)

    pad_idx = model.encoder.padding_idx
    eos_idx = model.encoder.eos_idx if hasattr(model.encoder, "eos_idx") else 2
    max_len = int(max_len_a * src_lengths.max().item() + max_len_b)

    beam_tokens = torch.full(
        (batch_size * beam_size, max_len + 1), pad_idx, dtype=torch.long, device=device
    )
    beam_tokens[:, 0] = eos_idx
    beam_scores = torch.zeros(batch_size, beam_size, device=device)
    beam_scores[:, 1:] = -1e9

    batch_dim = 0 if hasattr(model.encoder, "pos_encoder") else 1
    encoder_out_expanded = tuple(
        (
            eo.repeat_interleave(beam_size, dim=batch_dim)
            if eo is not None and eo.dim() > 1
            else (eo.repeat_interleave(beam_size) if eo is not None else None)
        )
        for eo in encoder_out
    )

    finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

    for step in range(1, max_len + 1):
        prev_tokens = beam_tokens[:, :step]
        logits, _ = model.decoder(prev_tokens, encoder_out_expanded)
        logits = logits[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        vocab_size = log_probs.size(-1)
        log_probs = log_probs.view(batch_size, beam_size, vocab_size)

        next_scores = beam_scores.unsqueeze(-1) + log_probs
        finished_mask = finished.unsqueeze(-1).expand_as(next_scores)
        next_scores.masked_fill_(finished_mask, -1e9)
        next_scores = next_scores.view(batch_size, -1)
        beam_scores, beam_idx = next_scores.topk(beam_size, dim=-1)

        beam_id = beam_idx // vocab_size
        token_id = beam_idx % vocab_size

        new_beam_tokens = beam_tokens.clone()
        for b in range(batch_size):
            for k in range(beam_size):
                src_beam = beam_id[b, k]
                new_beam_tokens[b * beam_size + k, :step] = beam_tokens[
                    b * beam_size + src_beam, :step
                ]
                new_beam_tokens[b * beam_size + k, step] = token_id[b, k]
        beam_tokens = new_beam_tokens

        finished = (token_id == eos_idx) | finished
        if finished.all():
            break

    results = []
    for b in range(batch_size):
        hypotheses = []
        for k in range(beam_size):
            tokens = beam_tokens[b * beam_size + k, 1 : step + 1]
            eos_positions = (tokens == eos_idx).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                tokens = tokens[: eos_positions[0]]
            hypotheses.append(
                {"tokens": tokens.cpu(), "score": beam_scores[b, k].item()}
            )
        results.append(hypotheses)
    return results


def decode(toks, dictionary, post_process="sentencepiece"):
    s = dictionary.string(toks.int().cpu(), post_process)
    return s if s else "<unk>"


def inference_step(sample, model, task, config, device):
    gen_out = beam_search_generate(
        model,
        sample,
        beam_size=config.beam,
        max_len_a=config.max_len_a,
        max_len_b=config.max_len_b,
        device=device,
    )
    srcs, hyps, refs = [], [], []
    for i in range(len(gen_out)):
        srcs.append(
            decode(
                strip_pad(sample["net_input"]["src_tokens"][i], task.src_dict.pad()),
                task.src_dict,
                config.post_process,
            )
        )
        hyps.append(decode(gen_out[i][0]["tokens"], task.tgt_dict, config.post_process))
        refs.append(
            decode(
                strip_pad(sample["target"][i], task.tgt_dict.pad()),
                task.tgt_dict,
                config.post_process,
            )
        )
    return srcs, hyps, refs


# ============================================================
# Training & Validation
# ============================================================


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def grouped_iterator(itr, n):
    group = []
    for item in itr:
        group.append(item)
        if len(group) == n:
            yield group
            group = []
    if group:
        yield group


def train_one_epoch(
    train_loader,
    model,
    task,
    criterion,
    optimizer,
    device,
    accum_steps=1,
    amp_dtype=None,
):
    itr = grouped_iterator(train_loader, accum_steps)
    total_steps = (len(train_loader) + accum_steps - 1) // accum_steps
    stats = {"loss": []}

    use_amp = amp_dtype is not None
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None

    model.train()
    progress = tqdm(itr, total=total_steps, desc="train", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0

        for i, sample in enumerate(samples):
            sample = move_to_device(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            if use_amp:
                with torch.amp.autocast(device.type, dtype=amp_dtype):
                    net_output = model.forward(**sample["net_input"])
                    lprobs = F.log_softmax(net_output[0], -1)
                    loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                    accum_loss += loss.item()
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            else:
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                accum_loss += loss.item()
                loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
            optimizer.multiply_grads(1 / (sample_size or 1.0))
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.multiply_grads(1 / (sample_size or 1.0))
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)

    loss_print = np.mean(stats["loss"])
    return stats


def validate(model, task, criterion, device, config, skip_bleu=False):
    valid_loader = load_data_iterator(
        task, "valid", 1, config.max_tokens, config.num_workers
    )
    stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
    srcs, hyps, refs = [], [], []

    model.eval()
    progress = tqdm(valid_loader, desc="validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            sample = move_to_device(sample, device=device)
            net_output = model.forward(**sample["net_input"])
            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = (
                criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                / sample_size
            )
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)

            if not skip_bleu:
                s, h, r = inference_step(sample, model, task, config, device)
                srcs.extend(s)
                hyps.extend(h)
                refs.extend(r)

    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    if not skip_bleu:
        tok = "zh" if task.tgt_lang == "zh" else "13a"
        stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
    else:
        stats["bleu"] = None
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    return stats


def validate_and_save(
    model, task, criterion, optimizer, epoch, config, device, save=True
):
    skip_bleu = epoch % config.validate_every != 0
    stats = validate(model, task, criterion, device, config, skip_bleu=skip_bleu)
    bleu = stats["bleu"]
    if bleu is None:
        return stats

    loss = stats["loss"]
    if save:
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)

        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step},
        }
        torch.save(check, savedir / f"checkpoint{epoch}.pt")
        shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / f"checkpoint_last.pt")

        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir / f"checkpoint_best.pt")

        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

    return stats


def try_load_checkpoint(model, optimizer=None, name=None, config=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir) / name
    if checkpath.exists():
        check = torch.load(checkpath, map_location="cpu")
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer is not None:
            optimizer._step = step = check["optim"]["step"]
        print(
            f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}"
        )
    else:
        print(f"no checkpoints found at {checkpath}")


def generate_prediction(
    model, task, config, device, split="test", outfile="./prediction.txt"
):
    task.load_dataset(split=split)
    pred_loader = load_data_iterator(
        task, split, 1, config.max_tokens, config.num_workers
    )
    idxs, hyps = [], []

    model.eval()
    progress = tqdm(pred_loader, desc="prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            sample = move_to_device(sample, device=device)
            s, h, r = inference_step(sample, model, task, config, device)
            hyps.extend(h)
            idxs.extend(sample["id"])

    hyps = [x for _, x in sorted(zip(idxs, hyps))]
    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h + "\n")
    print(f"predictions saved to {outfile}")


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="HW05 Seq2Seq Transformer 训练")
    parser.add_argument("--data-dir", default="./DATA/rawdata/ted2020")
    parser.add_argument("--save-dir", default="./checkpoints/rnn")
    parser.add_argument("--spm-model", default=None, help="sentencepiece model path")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=2.0)
    parser.add_argument("--lr-warmup", type=int, default=4000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=3)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--predict", action="store_true", help="仅做预测")
    parser.add_argument(
        "--no-bf16", action="store_true", help="NPU 上禁用 BF16，使用 FP16"
    )
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # 数据预处理（自动下载，已有则跳过）
    spm_model_path = prepare_data(
        data_dir=(
            args.data_dir.rsplit("/", 1)[0]
            if "/" in args.data_dir
            else "./DATA/rawdata"
        )
    )

    # 混合精度策略
    if device.type == "npu" and not args.no_bf16:
        amp_dtype = torch.bfloat16
        print("AMP: BF16 (升腾910)")
    elif device.type == "cuda":
        amp_dtype = torch.float16
        print("AMP: FP16 (CUDA)")
    else:
        amp_dtype = None
        print("AMP: 关闭")

    # 日志：同时输出到终端和文件
    log_file = Path(__file__).parent / f"train_{device.type}.log"
    logger = logging.getLogger("hw5.seq2seq")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    print(f"日志文件: {log_file}")
    logger.info(f"Device: {device}")
    logger.info(f"AMP dtype: {amp_dtype}")

    # 路径
    data_prefix = Path(args.data_dir)
    spm_model_path = spm_model_path or str(data_prefix / "spm8000.model")

    # 构建字典
    src_dict = build_dictionary(spm_model_path)
    tgt_dict = src_dict
    logger.info(f"Dictionary size: {len(src_dict)}")

    # Task
    task = TranslationTask(
        data_dir=str(data_prefix),
        src_lang="en",
        tgt_lang="zh",
        src_dict=src_dict,
        tgt_dict=tgt_dict,
    )

    # Config namespace (兼容 notebook 的 config 结构)
    from argparse import Namespace

    config = Namespace(
        savedir=args.save_dir,
        source_lang="en",
        target_lang="zh",
        max_tokens=args.max_tokens,
        accum_steps=args.accum_steps,
        num_workers=args.num_workers,
        max_epoch=args.max_epoch,
        start_epoch=1,
        beam=args.beam,
        max_len_a=1.2,
        max_len_b=10,
        post_process="sentencepiece",
        keep_last_epochs=5,
        validate_every=args.validate_every,
    )

    # 模型参数
    arch_args = Namespace(
        encoder_embed_dim=args.embed_dim,
        encoder_ffn_embed_dim=args.ffn_dim,
        encoder_layers=args.encoder_layers,
        encoder_attention_heads=args.nhead,
        decoder_embed_dim=args.embed_dim,
        decoder_ffn_embed_dim=args.ffn_dim,
        decoder_layers=args.decoder_layers,
        decoder_attention_heads=args.nhead,
        share_decoder_input_output_embed=True,
        dropout=args.dropout,
    )

    model = build_model(arch_args, task)
    model = model.to(device)
    logger.info(model)

    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.tgt_dict.pad(),
    ).to(device)

    optimizer = NoamOpt(
        model_size=arch_args.encoder_embed_dim,
        factor=args.lr_factor,
        warmup=args.lr_warmup,
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001
        ),
    )

    # 加载数据
    logger.info("loading data")
    task.load_dataset("train", max_samples=args.max_train_samples)
    task.load_dataset("valid", max_samples=args.max_valid_samples)

    # 仅预测模式
    if args.predict:
        try_load_checkpoint(model, name="avg_last_5_checkpoint.pt", config=config)
        generate_prediction(model, task, config, device)
        return

    # 恢复训练
    try_load_checkpoint(model, optimizer, name=args.resume, config=config)

    # 训练循环
    train_loader = load_data_iterator(
        task, "train", config.start_epoch, config.max_tokens, config.num_workers
    )
    for epoch in range(config.start_epoch, config.max_epoch + 1):
        train_one_epoch(
            train_loader,
            model,
            task,
            criterion,
            optimizer,
            device,
            config.accum_steps,
            amp_dtype,
        )
        stats = validate_and_save(
            model, task, criterion, optimizer, epoch, config, device
        )
        # 释放显存碎片，防止长时间训练 OOM
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'npu':
            torch.npu.empty_cache()
        if stats["bleu"] is not None:
            logger.info(
                f"epoch {epoch}: loss={stats['loss']:.4f} BLEU={stats['bleu'].score:.2f}"
            )
        else:
            logger.info(
                f"epoch {epoch}: loss={stats['loss']:.4f} (skip BLEU, next at epoch {epoch + config.validate_every})"
            )

    # 平均 checkpoint
    from copy import deepcopy

    checkdir = Path(config.savedir)
    checkpoints = sorted(checkdir.glob("checkpoint*.pt"))
    checkpoints = [
        c
        for c in checkpoints
        if "best" not in c.stem and "last" not in c.stem and "avg" not in c.stem
    ]
    checkpoints = checkpoints[-5:]
    if len(checkpoints) > 0:
        logger.info(f"Averaging {len(checkpoints)} checkpoints")
        avg_state = None
        for ckpt in checkpoints:
            state = torch.load(ckpt)
            if avg_state is None:
                avg_state = deepcopy(state)
            else:
                for k in avg_state["model"]:
                    avg_state["model"][k] += state["model"][k]
        for k in avg_state["model"]:
            avg_state["model"][k] /= len(checkpoints)
        output_path = checkdir / "avg_last_5_checkpoint.pt"
        torch.save(avg_state, output_path)
        logger.info(f"Saved averaged checkpoint: {output_path}")

    # 生成预测
    try_load_checkpoint(model, name="avg_last_5_checkpoint.pt", config=config)
    generate_prediction(model, task, config, device)


if __name__ == "__main__":
    main()
