import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("这个简化脚本只支持 NPU；当前环境没有可用 torch.npu。")
    return torch.device("npu")


def disable_mha_fastpath():
    mha_backend = getattr(torch.backends, "mha", None)
    set_fastpath_enabled = getattr(mha_backend, "set_fastpath_enabled", None)
    if set_fastpath_enabled is not None:
        set_fastpath_enabled(False)


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def rate(self):
        if self.step_num == 0:
            return 0.0
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(self.step_num ** (-0.5), self.step_num * self.warmup ** (-1.5))
        )

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_num": self.step_num,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_num = state["step_num"]


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, smoothing, ignore_index):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        lprobs = torch.log_softmax(logits, dim=-1)
        target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        pad_mask = target.eq(self.ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.smoothing / lprobs.size(-1)
        return (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss


def save_checkpoint(path, model, optimizer, epoch, valid_loss, bleu=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_loss": valid_loss,
            "bleu": bleu,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
