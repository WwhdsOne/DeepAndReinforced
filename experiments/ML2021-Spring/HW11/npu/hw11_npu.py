"""
HW11 DANN (Domain Adversarial Neural Network) - NPU 版本

单源域 DANN（Real → Drawing）+ 均衡推理（label balance）

参考 NTU ML2021Spring HW11 强基线（Arvin Liu）的方法重写：
  - ResNet18 backbone（保留我们已验证的 backbone，输出 512 维，兼容 Decoder）
  - loss_balance：训练时强制类别分布均匀（softmax(cls*10).mean ≈ uniform）
  - Decoder 自编码器（AE）：特征重建输入图 + denoising（uniform_noise p=0.02），重建损失 *3
  - source 随机 Canny + ×2 RandomErasing + RandomRotation(30)
  - target / test 不做 Canny（只 Grayscale + Resize），与目标域手绘涂鸦天然边缘化一致
  - 伪标签用 margin 筛选（top2[1]/top2[0] < thr & top1 ≥ prob_thre）+ 类反比加权，
    train_acc ≥ semi_acc 且 epoch ≥ min_dann_epochs 才启动；半监督阶段 source 真实标签
    + target 伪标签联合训练（防漂移 / 错误累积）
  - LR cosine 衰减到 eta_min=1e-6（≈0）
  - 训 500 epoch（参考：200 epoch 不稳，多训明显涨点）

用法:
  python hw11_npu.py                                  # 训练(DANN+AE+balance)+伪标签+推理 (全流程)
  python hw11_npu.py --mode train --epochs 500        # 仅训练(+伪标签)
  python hw11_npu.py --mode infer                     # 仅推理
  python hw11_npu.py --mode pseudo                    # 载入已训练权重，仅做伪标签微调+推理
  python hw11_npu.py --no-pseudo                      # 关闭伪标签自训练
"""

import argparse
import os
import random
import warnings
import numpy as np
import cv2
import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as NF
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.utils.data import DataLoader, ConcatDataset

warnings.filterwarnings(
    "ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*"
)

# ===================== 配置 =====================

LAMB_MAX = 1.0
GAMMA = 10.0

# 学习率（参考：F/AE 用 1e-4，C/D 用 1e-3；这里 C/D/AE 用 LR，FE 用 FE_LR 保护预训练 backbone）
LR = 1e-3
FE_LR = 1e-4

# 训练
NUM_EPOCHS = 500
BATCH_SIZE = 32
ETA_MIN = 1e-6          # cosine 末段 lr（≈0）
MIN_DANN_EPOCHS = 100   # DANN 至少先训满该 epoch 数，才允许切到半监督（防纯伪标签尾巴过长）

# DANN
ADV_SCALE = 1.0

# 损失权重（参考：balance 与 AE 均 *3）
BALANCE_WEIGHT = 3.0
AE_WEIGHT = 3.0

# 伪标签（margin 筛选）
PSEUDO_EPOCHS = 20
PSEUDO_THRESHOLD = 0.5       # top2[1] / top2[0] < 0.5
PSEUDO_PROB_THRE = 0.3       # top1 >= 0.3
SEMI_ACC = 0.85              # train_acc 达到后才启动半监督

# 路径
EXTRACTOR_PATH = "extractor_model.bin"
PREDICTOR_PATH = "predictor_model.bin"
DISCRIMINATOR_PATH = "discriminator_model.bin"
AE_PATH = "ae_model.bin"
SOURCE_DIR = "../real_or_drawing/train_data"
TARGET_DIR = "../real_or_drawing/test_data"


# ===================== 数据变换 =====================
# source：随机 Canny（让照片更像手绘涂鸦）+ 两次 RandomErasing + 随机翻转/旋转
# target / test：不做 Canny（手绘已是天然边缘图），只 Grayscale + Resize

source_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Lambda(
            lambda x: cv2.Canny(
                np.array(x),
                np.random.randint(170, 200),
                np.random.randint(250, 300),
            )
        ),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30, fill=(0,)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.03, 0.07), ratio=(0.3, 3), value=1 - 1e-6),
        transforms.RandomErasing(p=0.25, scale=(0.03, 0.07), ratio=(0.3, 3), value=1e-6),
    ]
)

target_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30, fill=(0,)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.03, 0.07), ratio=(0.3, 3), value=1 - 1e-6),
        transforms.RandomErasing(p=0.25, scale=(0.03, 0.07), ratio=(0.3, 3), value=1e-6),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)


# ===================== 模型定义 =====================


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights="DEFAULT")
        old_weight = backbone.conv1.weight.clone()  # [64, 3, 7, 7]
        # CIFAR 风格：3x3 首层 + 去掉 MaxPool，适配 32x32 输入
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        with torch.no_grad():
            # RGB 三通道平均 → 单通道，7x7 中心 3x3 裁剪
            backbone.conv1.weight.copy_(
                old_weight.mean(1, keepdim=True)[:, :, 2:5, 2:5]
            )
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class LabelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        return self.layer(h)


class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )

    def forward(self, h):
        return self.layer(h)


class Decoder(nn.Module):
    """自编码器：将 512 维特征重建为 32x32 单通道图（denoising 训练）。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv(x.view(-1, 512, 1, 1))
        return out


# ===================== 调度函数 =====================


def adaptive_lambda(step, total_steps, gamma=GAMMA):
    """DANN 论文的自适应 lambda: 从 0 渐增到 LAMB_MAX"""
    p = step / total_steps
    return LAMB_MAX * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


def uniform_noise(x, p=0.025):
    """denoising AE：随机把少量像素推向两端，逼迫 AE 学结构性特征。"""
    out = x.clone()
    noise = torch.rand_like(out)
    out[noise <= p] = 1e-6
    out[noise >= (1 - p)] = 1 - 1e-6
    return out


# ===================== 均衡推理 =====================


def balanced_predict(probs, num_classes=10):
    """基于置信度的均衡分配：每类恰好 N/10 个样本（输入须为已 softmax 的概率）"""
    n = probs.size(0)
    per_class = n // num_classes

    probs_np = probs.numpy()
    predictions = np.full(n, -1, dtype=np.int64)
    quota = np.full(num_classes, per_class, dtype=np.int64)

    flat_probs = probs_np.flatten()
    sorted_indices = np.argsort(-flat_probs)

    for idx in sorted_indices:
        sample_id = idx // num_classes
        class_id = idx % num_classes
        if predictions[sample_id] == -1 and quota[class_id] > 0:
            predictions[sample_id] = class_id
            quota[class_id] -= 1

    unassigned = np.where(predictions == -1)[0]
    if len(unassigned) > 0:
        predictions[unassigned] = probs_np[unassigned].argmax(axis=1)

    return torch.from_numpy(predictions)


# ===================== 训练循环 =====================


def train_epoch(
    source_loader,
    target_loader,
    models,
    optimizers,
    criterions,
    scaler,
    adv_scale,
    global_step,
    total_steps,
    uniform_dist,
    balance_weight=3.0,
    ae_weight=3.0,
    prefix="",
):
    feature_extractor, label_predictor, domain_classifier, decoder = models
    optimizer_F, optimizer_C, optimizer_D, optimizer_AE = optimizers
    class_criterion, domain_criterion, ae_criterion, balance_criterion = criterions

    running_D_loss, running_F_loss = 0.0, 0.0
    running_AE_loss, running_B_loss = 0.0, 0.0
    running_cls, running_dom = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(
        tqdm(zip(source_loader, target_loader), desc=prefix, leave=True)
    ):
        lamb = adaptive_lambda(global_step + i, total_steps)
        source_data = source_data.npu()
        source_label = source_label.npu()
        target_data = target_data.npu()
        n_source = source_data.shape[0]

        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros(mixed_data.shape[0], 1).npu()
        domain_label[:n_source] = 1

        # Step 1: Domain Classifier
        optimizer_D.zero_grad()
        with torch.autocast("npu"):
            feature = feature_extractor(mixed_data)
            domain_logits = domain_classifier(feature.detach())
            loss_D = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss_D.item()
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Step 2: Feature Extractor + Label Predictor + Decoder(AE)
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        optimizer_AE.zero_grad()
        with torch.autocast("npu"):
            feature = feature_extractor(mixed_data)
            class_logits = label_predictor(feature[:n_source])
            domain_logits = domain_classifier(feature)

            # 重建损失（denoising）：对加噪输入重建干净图
            recon = decoder(feature_extractor(uniform_noise(mixed_data, 0.02)))
            loss_ae = ae_criterion(recon, mixed_data)

            # 类别分布均衡约束
            loss_balance = balance_criterion(
                NF.softmax(class_logits * 10, 1).mean(0), uniform_dist
            )

            loss_cls = class_criterion(class_logits, source_label)
            loss_domain = domain_criterion(domain_logits, domain_label)
            loss_F = (
                loss_cls
                - adv_scale * lamb * loss_domain
                + balance_weight * loss_balance
                + ae_weight * loss_ae
            )
        running_F_loss += loss_F.item()
        running_AE_loss += loss_ae.item()
        running_B_loss += loss_balance.item()
        running_cls += loss_cls.item()
        running_dom += loss_domain.item()
        scaler.scale(loss_F).backward()
        scaler.step(optimizer_C)
        scaler.step(optimizer_F)
        scaler.step(optimizer_AE)
        scaler.update()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += n_source

        n = i + 1
    return {
        "D_loss": running_D_loss / n,
        "F_loss": running_F_loss / n,
        "AE_loss": running_AE_loss / n,
        "B_loss": running_B_loss / n,
        "cls": running_cls / n,
        "domain": running_dom / n,
        "acc": total_hit / total_num,
        "step": global_step + n,
    }


def train_cls_epoch(
    loader,
    models,
    optimizers,
    scaler,
    class_sep_criterion,
    balance_criterion,
    uniform_dist,
    weight,
    balance_weight=3.0,
    prefix="",
):
    """半监督阶段：在 source 真实标签 + target 伪标签 联合数据集上微调，
    带类反比加权（weight[ans]）+ 均衡约束，防漂移 / 错误累积。"""
    feature_extractor, label_predictor, _, _ = models
    optimizer_F, optimizer_C, _, _ = optimizers

    total_hit, total_num, running_loss, steps = 0.0, 0.0, 0.0, 0
    for i, (data, ans) in enumerate(tqdm(loader, desc=prefix, leave=True)):
        data, ans = data.npu(), ans.npu()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        with torch.autocast("npu"):
            class_logits = label_predictor(feature_extractor(data))
            loss_balance = balance_criterion(
                NF.softmax(class_logits * 10, 1).mean(0), uniform_dist
            )
            loss_sep = class_sep_criterion(class_logits, ans)
            loss = (weight[ans] * loss_sep).mean() + balance_weight * loss_balance
        running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer_C)
        scaler.step(optimizer_F)
        scaler.update()
        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == ans).item()
        total_num += data.shape[0]
        steps += 1
    return running_loss / max(1, steps), total_hit / max(1, total_num)


# ===================== 伪标签生成（margin 筛选） =====================


class SubsetCustomLabel:
    def __init__(self, dataset, labels, indices):
        self.dataset = dataset
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]][0], self.labels[idx]


def generate_pseudo_labels(feature_extractor, label_predictor, target_dataset, batch_size=128):
    """margin 筛选高置信目标域样本：top2[1]/top2[0] < thr 且 top1 ≥ prob_thre。

    返回 (SubsetCustomLabel, count) —— count 为各类预测概率之和，用于类反比加权。
    不使用目标域真实标签，完全由模型生成。
    """
    feature_extractor.eval()
    label_predictor.eval()
    softmax = nn.Softmax(dim=-1)
    idx, targets, count = [], [], torch.zeros(10)

    pseudo_dataset = ImageFolder(target_dataset.root, transform=test_transform)
    loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        for i, (img, _) in enumerate(loader):
            logits = label_predictor(feature_extractor(img.npu()))
            probs = softmax(logits)
            top2 = probs.topk(2, dim=1).values
            ratio = top2[:, 1] / top2[:, 0]
            select = (ratio < PSEUDO_THRESHOLD) & (top2[:, 0] >= PSEUDO_PROB_THRE)
            count += probs[select].sum(0).cpu()
            if not select.any():
                continue
            targets.append(probs[select].argmax(dim=1))
            idx += (torch.where(select)[0] + batch_size * i).tolist()

    if len(targets) == 0:
        return None, None
    targets = torch.cat(targets, dim=0).cpu().tolist()
    new = SubsetCustomLabel(pseudo_dataset, targets, idx)
    feature_extractor.train()
    label_predictor.train()
    return new, count


# ===================== 推理 =====================


def run_inference(feature_extractor, label_predictor, test_loader):
    feature_extractor.load_state_dict(
        torch.load(EXTRACTOR_PATH, map_location="npu:0", weights_only=False)
    )
    label_predictor.load_state_dict(
        torch.load(PREDICTOR_PATH, map_location="npu:0", weights_only=False)
    )
    feature_extractor.eval()
    label_predictor.eval()

    all_probs = []
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(test_loader, desc="[infer]")):
            logits = label_predictor(feature_extractor(data.npu())).cpu()
            all_probs.append(torch.softmax(logits, dim=1))

    print()
    all_probs = torch.cat(all_probs, dim=0)
    print("  均衡分配中...")
    predictions = balanced_predict(all_probs)

    df = pd.DataFrame({"id": np.arange(len(predictions)), "label": predictions.numpy()})
    df.to_csv("DaNN_submission.csv", index=False)
    print(f"均衡推理完成，结果已保存到 DaNN_submission.csv ({len(predictions)} 条)")


# ===================== 保存/加载 =====================


def save_models(feature_extractor, label_predictor, domain_classifier, decoder):
    torch.save(feature_extractor.state_dict(), EXTRACTOR_PATH)
    torch.save(label_predictor.state_dict(), PREDICTOR_PATH)
    torch.save(domain_classifier.state_dict(), DISCRIMINATOR_PATH)
    torch.save(decoder.state_dict(), AE_PATH)
    print(f"\n模型已保存: {EXTRACTOR_PATH}, {PREDICTOR_PATH}, {AE_PATH}")


# ===================== 主函数 =====================


def main():
    parser = argparse.ArgumentParser(description="HW11 DANN 域对抗训练（参考强基线重写）")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "all", "pseudo"],
        default="all",
        help="train=训练, infer=推理, all=全流程, pseudo=载入已训练权重仅做伪标签微调 (默认 all)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lamb-max", type=float, default=LAMB_MAX)
    parser.add_argument("--adv-scale", type=float, default=ADV_SCALE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--fe-lr", type=float, default=FE_LR)
    parser.add_argument("--balance-weight", type=float, default=BALANCE_WEIGHT)
    parser.add_argument("--ae-weight", type=float, default=AE_WEIGHT)
    parser.add_argument(
        "--pseudo", action="store_true", default=True, help="伪标签半监督自训练"
    )
    parser.add_argument("--no-pseudo", dest="pseudo", action="store_false")
    parser.add_argument("--pseudo-epochs", type=int, default=PSEUDO_EPOCHS)
    parser.add_argument("--pseudo-threshold", type=float, default=PSEUDO_THRESHOLD)
    parser.add_argument("--pseudo-prob-thre", type=float, default=PSEUDO_PROB_THRE)
    parser.add_argument("--semi-acc", type=float, default=SEMI_ACC)
    parser.add_argument(
        "--min-dann-epochs",
        type=int,
        default=MIN_DANN_EPOCHS,
        help="DANN 至少先训满该 epoch 数，才允许切到半监督（防纯伪标签尾巴过长）",
    )
    args = parser.parse_args()

    # ---- 数据加载 ----
    source_dataset = ImageFolder(SOURCE_DIR, transform=source_transform)
    source_loader = DataLoader(
        source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    print(f"  源域 (real): {len(source_dataset)} 张")

    target_dataset = ImageFolder(TARGET_DIR, transform=target_transform)
    target_loader = DataLoader(
        target_dataset, batch_size=args.batch_size * 4, shuffle=True, num_workers=2
    )
    test_dataset = ImageFolder(TARGET_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2
    )
    print(f"  目标域 (drawing): {len(target_dataset)} 张")

    # ---- 模型初始化 ----
    feature_extractor = FeatureExtractor().npu()
    label_predictor = LabelPredictor().npu()
    domain_classifier = DomainClassifier().npu()
    decoder = Decoder().npu()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    ae_criterion = nn.MSELoss()
    balance_criterion = nn.MSELoss()
    class_sep_criterion = nn.CrossEntropyLoss(reduction="none")
    uniform_dist = torch.full((10,), 1.0 / 10).npu()

    optimizer_F = optim.AdamW(
        feature_extractor.parameters(), lr=args.fe_lr, weight_decay=1e-4
    )
    optimizer_C = optim.AdamW(
        label_predictor.parameters(), lr=args.lr, weight_decay=1e-3
    )
    optimizer_D = optim.AdamW(
        domain_classifier.parameters(), lr=args.lr, weight_decay=1e-3
    )
    optimizer_AE = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("npu")

    models = (feature_extractor, label_predictor, domain_classifier, decoder)
    optimizers = (optimizer_F, optimizer_C, optimizer_D, optimizer_AE)
    criterions = (class_criterion, domain_criterion, ae_criterion, balance_criterion)

    # 4 个优化器共用 cosine 调度（末段 eta_min≈0），每个 epoch 步进一步
    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs, eta_min=ETA_MIN)
        for o in optimizers
    ]

    # ---- 仅伪标签微调：载入已训练权重，跳过 DANN 训练 ----
    if args.mode == "pseudo":
        feature_extractor.load_state_dict(
            torch.load(EXTRACTOR_PATH, map_location="npu:0", weights_only=False)
        )
        label_predictor.load_state_dict(
            torch.load(PREDICTOR_PATH, map_location="npu:0", weights_only=False)
        )
        decoder.load_state_dict(
            torch.load(AE_PATH, map_location="npu:0", weights_only=False)
        )
        domain_classifier.load_state_dict(
            torch.load(DISCRIMINATOR_PATH, map_location="npu:0", weights_only=False)
        )
        print("  已载入已训练模型，跳过 DANN 训练，直接进入伪标签微调")

    # ---- Phase 1: DANN 训练（+AE + balance），acc 达标后切伪标签 ----
    if args.mode in ("train", "all"):
        print(f"\n=== DANN 训练 ({args.epochs} epochs) ===")
        print(f"  lambda: 0 → {args.lamb_max} (自适应), adv_scale={args.adv_scale}")
        print(
            f"  lr: FE={args.fe_lr}, C/D/AE={args.lr}, cosine→{ETA_MIN} (≈0), "
            f"balance×{args.balance_weight}, ae×{args.ae_weight}"
        )
        global_step = 0
        train_acc = 0.0
        semi_flg = False
        try:
            for epoch in range(args.epochs):
                can_semi = (
                    epoch >= args.min_dann_epochs
                    and (train_acc >= args.semi_acc or semi_flg)
                    and args.pseudo
                )
                if can_semi:
                    pseudo_dataset, count = generate_pseudo_labels(
                        feature_extractor, label_predictor, target_dataset
                    )
                    if pseudo_dataset is not None:
                        semi_flg = True
                        weight = (count.mean() / (count + 1e-6)).npu()
                        print(
                            f"  Get pseudo label: {len(pseudo_dataset)} | "
                            + ", ".join(f"{int(c):4d}" for c in count.tolist())
                        )
                        # 半监督阶段：source 真实标签 + target 伪标签 联合训练，防漂移/错误累积
                        combined = ConcatDataset([source_dataset, pseudo_dataset])
                        semi_loader = DataLoader(
                            combined,
                            batch_size=args.batch_size * 5,
                            shuffle=True,
                            num_workers=2,
                        )
                        loss, acc = train_cls_epoch(
                            semi_loader,
                            models,
                            optimizers,
                            scaler,
                            class_sep_criterion,
                            balance_criterion,
                            uniform_dist,
                            weight,
                            balance_weight=args.balance_weight,
                            prefix=f"Epoch {epoch+1}/{args.epochs} [semi] ",
                        )
                        print(
                            f"epoch {epoch:>3d}: semi loss={loss:.4f} acc={acc:.4f}"
                        )
                    else:
                        # 伪标签为空，退回 DANN 训练本 epoch
                        stats = train_epoch(
                            source_loader,
                            target_loader,
                            models,
                            optimizers,
                            criterions,
                            scaler,
                            args.adv_scale,
                            global_step,
                            args.epochs * min(len(source_loader), len(target_loader)),
                            uniform_dist,
                            balance_weight=args.balance_weight,
                            ae_weight=args.ae_weight,
                        )
                        global_step = stats["step"]
                        train_acc = stats["acc"]
                        current_lamb = adaptive_lambda(global_step, args.epochs * min(len(source_loader), len(target_loader)))
                        print(
                            f"epoch {epoch:>3d}: D={stats['D_loss']:.4f} F={stats['F_loss']:.4f} "
                            f"AE={stats['AE_loss']:.4f} bal={stats['B_loss']:.4f} "
                            f"cls={stats['cls']:.3f} dom={stats['domain']:.3f} "
                            f"acc={stats['acc']:.4f} λ={current_lamb:.3f}"
                        )
                else:
                    stats = train_epoch(
                        source_loader,
                        target_loader,
                        models,
                        optimizers,
                        criterions,
                        scaler,
                        args.adv_scale,
                        global_step,
                        args.epochs * min(len(source_loader), len(target_loader)),
                        uniform_dist,
                        balance_weight=args.balance_weight,
                        ae_weight=args.ae_weight,
                        prefix=f"Epoch {epoch+1}/{args.epochs} [DANN] ",
                    )
                    global_step = stats["step"]
                    train_acc = stats["acc"]
                    current_lamb = adaptive_lambda(global_step, args.epochs * min(len(source_loader), len(target_loader)))
                    print(
                        f"epoch {epoch:>3d}: D={stats['D_loss']:.4f} F={stats['F_loss']:.4f} "
                        f"AE={stats['AE_loss']:.4f} bal={stats['B_loss']:.4f} "
                        f"cls={stats['cls']:.3f} dom={stats['domain']:.3f} "
                        f"acc={stats['acc']:.4f} λ={current_lamb:.3f}"
                    )
                save_models(feature_extractor, label_predictor, domain_classifier, decoder)
                for sch in schedulers:
                    sch.step()
        except KeyboardInterrupt:
            print("\n训练中断，保存当前模型...")
            save_models(feature_extractor, label_predictor, domain_classifier, decoder)

    # ---- Phase 1.5（仅 pseudo 模式）：伪标签微调若干 epoch ----
    if args.mode == "pseudo" and args.pseudo:
        print("\n=== 伪标签微调 ===")
        for epoch in range(args.pseudo_epochs):
            pseudo_dataset, count = generate_pseudo_labels(
                feature_extractor, label_predictor, target_dataset
            )
            if pseudo_dataset is None:
                print("  无合格伪标签，跳过")
                break
            weight = (count.mean() / (count + 1e-6)).npu()
            pseudo_loader = DataLoader(
                pseudo_dataset,
                batch_size=args.batch_size * 5,
                shuffle=True,
                num_workers=2,
            )
            loss, acc = train_cls_epoch(
                pseudo_loader,
                models,
                optimizers,
                scaler,
                class_sep_criterion,
                balance_criterion,
                uniform_dist,
                weight,
                balance_weight=args.balance_weight,
                prefix=f"pseudo {epoch+1}/{args.pseudo_epochs} ",
            )
            print(f"  pseudo epoch {epoch+1}/{args.pseudo_epochs}: loss={loss:.4f} acc={acc:.4f}")
            save_models(feature_extractor, label_predictor, domain_classifier, decoder)
            for sch in schedulers:
                sch.step()

    # ---- Phase 2: 均衡推理 ----
    if args.mode in ("infer", "all", "pseudo"):
        print("\n=== 均衡推理 ===")
        run_inference(feature_extractor, label_predictor, test_loader)


if __name__ == "__main__":
    main()
