#!/usr/bin/env python3
"""HW05 NPU-only 简化训练脚本。

前提：数据已经预处理好，目录内存在 train.en/train.zh/valid.en/valid.zh/spm8000.model。
"""

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .data import build_dataloader, build_dictionary, load_parallel_dataset
from .generate import greedy_decode
from .model import TransformerSeq2Seq
from .train_utils import (
    LabelSmoothedCrossEntropy,
    NoamOpt,
    disable_mha_fastpath,
    load_checkpoint,
    move_to_device,
    require_npu,
    save_checkpoint,
    set_seed,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HW05 NPU-only Transformer 训练")
    parser.add_argument("--data-dir", default="./DATA/rawdata/ted2020")
    parser.add_argument("--save-dir", default="./checkpoints/npu-simple")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-src-len", type=int, default=160)
    parser.add_argument("--max-tgt-len", type=int, default=160)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=2.0)
    parser.add_argument("--lr-warmup", type=int, default=4000)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--eval-bleu", action="store_true")
    parser.add_argument("--max-bleu-samples", type=int, default=200)
    parser.add_argument("--max-decode-len", type=int, default=120)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args(argv)


def setup_logger():
    logger = logging.getLogger("hw05.npu_simple")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def build_model_args(args):
    return Namespace(
        embed_dim=args.embed_dim,
        ffn_dim=args.ffn_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    )


def train_one_epoch(loader, model, criterion, optimizer, device, accum_steps, amp_dtype):
    model.train()
    losses = []
    optimizer.zero_grad()
    progress = tqdm(loader, desc="train", leave=False)
    for step, sample in enumerate(progress, start=1):
        sample = move_to_device(sample, device)
        with torch.amp.autocast(device.type, dtype=amp_dtype):
            logits = model(**sample["net_input"])
            loss = criterion(logits, sample["target"]) / sample["ntokens"]
            scaled_loss = loss / accum_steps
        scaled_loss.backward()

        if step % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        losses.append(float(loss.item()))
        progress.set_postfix(loss=losses[-1], lr=optimizer.rate())

    if len(loader) % accum_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def validate_loss(loader, model, criterion, device, amp_dtype):
    model.eval()
    losses = []
    progress = tqdm(loader, desc="valid-loss", leave=False)
    for sample in progress:
        sample = move_to_device(sample, device)
        with torch.amp.autocast(device.type, dtype=amp_dtype):
            logits = model(**sample["net_input"])
            loss = criterion(logits, sample["target"]) / sample["ntokens"]
        losses.append(float(loss.item()))
        progress.set_postfix(loss=losses[-1])
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def validate_bleu(loader, model, dictionary, device, args):
    hyps = []
    refs = []
    progress = tqdm(loader, desc="greedy-bleu", leave=False)
    for sample in progress:
        sample = move_to_device(sample, device)
        hypotheses = greedy_decode(
            model,
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            max_len=args.max_decode_len,
            bos_idx=dictionary.eos(),
            eos_idx=dictionary.eos(),
        )
        for hypothesis, target in zip(hypotheses, sample["target"].cpu()):
            hyps.append(dictionary.string(hypothesis))
            refs.append(dictionary.string(target))
            if args.max_bleu_samples > 0 and len(hyps) >= args.max_bleu_samples:
                break
        progress.set_postfix(samples=len(hyps))
        if args.max_bleu_samples > 0 and len(hyps) >= args.max_bleu_samples:
            break
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score


def main(argv=None):
    args = parse_args(argv)
    logger = setup_logger()
    set_seed(args.seed)
    device = require_npu()
    disable_mha_fastpath()
    amp_dtype = torch.bfloat16

    data_dir = Path(args.data_dir)
    spm_model = data_dir / "spm8000.model"
    dictionary = build_dictionary(spm_model)
    logger.info("dictionary size=%d", len(dictionary))

    train_dataset = load_parallel_dataset(
        data_dir,
        "train",
        dictionary,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )
    valid_dataset = load_parallel_dataset(
        data_dir,
        "valid",
        dictionary,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )
    logger.info(
        "train=%d filtered=%d | valid=%d filtered=%d",
        len(train_dataset),
        train_dataset.filtered_count,
        len(valid_dataset),
        valid_dataset.filtered_count,
    )

    model = TransformerSeq2Seq(build_model_args(args), dictionary, dictionary).to(device)
    criterion = LabelSmoothedCrossEntropy(0.1, ignore_index=dictionary.pad()).to(device)
    optimizer = NoamOpt(
        model_size=args.embed_dim,
        factor=args.lr_factor,
        warmup=args.lr_warmup,
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=0.0,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.0001,
        ),
    )

    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("resume from %s epoch=%d", args.resume, checkpoint["epoch"])

    best_loss = float("inf")
    save_dir = Path(args.save_dir)
    for epoch in range(start_epoch, args.max_epoch + 1):
        train_loader = build_dataloader(
            train_dataset,
            dictionary,
            max_tokens=args.max_tokens,
            epoch=epoch,
            shuffle=True,
            num_workers=args.num_workers,
        )
        valid_loader = build_dataloader(
            valid_dataset,
            dictionary,
            max_tokens=args.max_tokens,
            epoch=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            args.accum_steps,
            amp_dtype,
        )
        valid_loss = validate_loss(valid_loader, model, criterion, device, amp_dtype)
        bleu = None
        if args.eval_bleu and epoch % args.validate_every == 0:
            bleu = validate_bleu(valid_loader, model, dictionary, device, args)

        save_checkpoint(
            save_dir / "checkpoint_last.pt",
            model,
            optimizer,
            epoch,
            valid_loss,
            bleu=bleu,
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(
                save_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                valid_loss,
                bleu=bleu,
            )

        if device.type == "npu":
            torch.npu.empty_cache()
        logger.info(
            "epoch %d: train_loss=%.4f valid_loss=%.4f bleu=%s",
            epoch,
            train_loss,
            valid_loss,
            "skip" if bleu is None else f"{bleu:.2f}",
        )


if __name__ == "__main__":
    main()
