"""
在 Tiny ImageNet-200 上训练模型的通用脚本。

用法：
    python train.py --model VGG16 --epochs 30 --batch-size 128
    python train.py --model VGG19 --epochs 50 --lr 0.001
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 将 ch8 加入路径，方便直接 python train.py 运行
sys.path.insert(0, str(Path(__file__).parent))
from tiny_imagenet import get_loaders
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny ImageNet-200 训练")
    parser.add_argument("--model", default="VGG16",
                        help="模型名称，如 VGG16, VGG19 (默认 VGG16)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--data-root", default="../data",
                        help="包含 tiny-imagenet-200 的目录 (默认 ../data)")
    parser.add_argument("--device", default="auto",
                        help="训练设备: cuda, cpu, auto (默认 auto)")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="  train", leave=False, ncols=80)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        # 实时更新进度条后缀
        acc = 100. * correct / total_samples
        pbar.set_postfix(loss=loss.item(), acc=f"{acc:.1f}%")

    return total_loss / len(loader), 100. * correct / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    pbar = tqdm(loader, desc="  val", leave=False, ncols=80)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader), 100. * correct / len(loader.dataset)


def main():
    args = parse_args()

    # ── 设备 ─────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")
    print(f"模型: {args.model}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | "
          f"LR: {args.lr}")

    # ── 数据 ─────────────────────────────────────
    print("\n正在加载 Tiny ImageNet-200...")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size)
    print(f"训练集: {len(train_loader.dataset)} 张 | "
          f"验证集: {len(val_loader.dataset)} 张")

    # ── 模型 ─────────────────────────────────────
    model = get_model(args.model, num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5
    )

    # ── 训练循环 ─────────────────────────────────
    print(f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Train Acc':>10}  "
          f"{'Val Loss':>9}  {'Val Acc':>9}  {'Time':>6}")
    print("-" * 60)

    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        marker = " *" if val_acc > best_acc else ""
        if val_acc > best_acc:
            best_acc = val_acc

        print(f"{epoch:5d}  {train_loss:11.4f}  {train_acc:9.2f}%  "
              f"{val_loss:9.4f}  {val_acc:8.2f}%  {epoch_time:5.0f}s{marker}")

    # ── 保存 ─────────────────────────────────────
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / f"{args.model}_tiny_imagenet.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存: {model_path}")

    history_path = artifacts_dir / f"{args.model}_tiny_imagenet_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"训练记录已保存: {history_path}")
    print(f"最佳验证准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
