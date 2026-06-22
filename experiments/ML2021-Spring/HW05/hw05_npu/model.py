import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(1)].unsqueeze(0))


class TransformerSeq2Seq(nn.Module):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__()
        self.src_pad_idx = src_dict.pad()
        self.tgt_pad_idx = tgt_dict.pad()
        self.src_embed = nn.Embedding(len(src_dict), args.embed_dim, self.src_pad_idx)
        self.tgt_embed = nn.Embedding(len(tgt_dict), args.embed_dim, self.tgt_pad_idx)
        self.pos_encoder = PositionalEncoding(args.embed_dim, args.dropout)
        self.pos_decoder = PositionalEncoding(args.embed_dim, args.dropout)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=args.embed_dim,
                    nhead=args.nhead,
                    dim_feedforward=args.ffn_dim,
                    dropout=args.dropout,
                    batch_first=True,
                )
                for _ in range(args.encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=args.embed_dim,
                    nhead=args.nhead,
                    dim_feedforward=args.ffn_dim,
                    dropout=args.dropout,
                    batch_first=True,
                )
                for _ in range(args.decoder_layers)
            ]
        )
        self.output_projection = nn.Linear(args.embed_dim, len(tgt_dict), bias=False)
        self.output_projection.weight = self.tgt_embed.weight
        self.apply(self._init_parameters)

    def _init_parameters(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def encode(self, src_tokens):
        x = self.pos_encoder(self.src_embed(src_tokens))
        src_key_padding_mask = src_tokens.eq(self.src_pad_idx)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask

    def decode(self, prev_output_tokens, encoder_outputs, src_key_padding_mask):
        x = self.pos_decoder(self.tgt_embed(prev_output_tokens))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_outputs,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
        return self.output_projection(x)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        del src_lengths
        encoder_outputs, src_key_padding_mask = self.encode(src_tokens)
        logits = self.decode(prev_output_tokens, encoder_outputs, src_key_padding_mask)
        return logits
