#!/usr/bin/env python3
"""HW05 NPU-only 简化推理脚本。"""

import argparse
from argparse import Namespace
from pathlib import Path

import sentencepiece as spm
import torch

from .data import build_dictionary
from .generate import greedy_decode
from .model import TransformerSeq2Seq
from .train_utils import disable_mha_fastpath, load_checkpoint, require_npu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HW05 NPU-only Transformer 推理")
    parser.add_argument("--data-dir", default="./DATA/rawdata/ted2020")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-decode-len", type=int, default=120)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args(argv)


def build_model_args(args):
    return Namespace(
        embed_dim=args.embed_dim,
        ffn_dim=args.ffn_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    )


def encode_text(text, sp_processor, dictionary):
    pieces = sp_processor.encode(text.strip(), out_type=str)
    token_ids = dictionary.encode(pieces) + [dictionary.eos()]
    src_tokens = torch.tensor([token_ids], dtype=torch.long)
    src_lengths = torch.tensor([len(token_ids)], dtype=torch.long)
    return src_tokens, src_lengths


def main(argv=None):
    args = parse_args(argv)
    device = require_npu()
    disable_mha_fastpath()

    data_dir = Path(args.data_dir)
    spm_model = data_dir / "spm8000.model"
    dictionary = build_dictionary(spm_model)
    sp_processor = spm.SentencePieceProcessor(model_file=str(spm_model))

    model = TransformerSeq2Seq(build_model_args(args), dictionary, dictionary).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    src_tokens, src_lengths = encode_text(args.text, sp_processor, dictionary)
    src_tokens = src_tokens.to(device)
    src_lengths = src_lengths.to(device)
    hypotheses = greedy_decode(
        model,
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        max_len=args.max_decode_len,
        bos_idx=dictionary.eos(),
        eos_idx=dictionary.eos(),
    )
    print(dictionary.string(hypotheses[0]))


if __name__ == "__main__":
    main()
