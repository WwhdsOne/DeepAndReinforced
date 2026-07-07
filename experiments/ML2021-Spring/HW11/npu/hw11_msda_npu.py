"""
HW11 MSDA (Multi-Source Domain Adaptation) - NPU 版本

四源域 (Sobel/Canny/Laplacian/Sketch) + 矩匹配 + 伪标签 + 均衡推理

用法:
  python hw11_preprocess.py                          # 先预处理
  python hw11_msda_npu.py                            # 训练+推理 (全流程)
  python hw11_msda_npu.py --mode train --epochs 200  # 仅训练
  python hw11_msda_npu.py --mode pseudo              # 伪标签微调
  python hw11_msda_npu.py --mode infer               # 仅推理
"""

import argparse
import random
import numpy as np
import cv2
import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd

# ===================== 配置 =====================

LAMB_MAX = 1.0
GAMMA = 10.0
LR = 1e-3
WARMUP_RATIO = 0.1
MM_WEIGHT = 0.01
NUM_EPOCHS = 200
PSEUDO_EPOCHS = 50
PSEUDO_THRESHOLD = 0.95
EXTRACTOR_PATH = 'extractor_model.bin'
PREDICTOR_PATH = 'predictor_model.bin'
SOURCE_DOMAINS = ['sobel', 'canny', 'laplacian', 'sketch']
MSDA_DATA_DIR = '../real_or_drawing/msda_data'
TARGET_DIR = '../real_or_drawing/test_data'

# ===================== 数据变换 =====================

source_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# ===================== 多源域数据集 =====================

class MultiSourceDataset(Dataset):
    """从多个源域中均匀采样"""

    def __init__(self, datasets):
        self.datasets = datasets
        self.offsets = [0]
        for ds in datasets:
            self.offsets.append(self.offsets[-1] + len(ds))
        self.total = self.offsets[-1]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            if start <= idx < end:
                img, label = self.datasets[i][idx - start]
                return img, label
        raise IndexError(idx)


# ===================== 模型定义 =====================

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x).squeeze()


class LabelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        return self.layer(h)


class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, h):
        return self.layer(h)


# ===================== 调度函数 =====================

def adaptive_lambda(step, total_steps, gamma=GAMMA):
    """DANN 论文的自适应 lambda: 从 0 渐增到 LAMB_MAX"""
    p = step / total_steps
    return LAMB_MAX * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


def create_lr_scheduler(optimizer, total_steps, warmup_ratio=WARMUP_RATIO):
    """Warmup + Cosine Annealing"""
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================== 矩匹配损失 =====================

def moment_matching_loss(source_features, target_features):
    """一阶（均值）和二阶（协方差）矩匹配"""
    s_mean = source_features.mean(dim=0)
    t_mean = target_features.mean(dim=0)
    loss = (s_mean - t_mean).pow(2).sum()

    if source_features.size(0) > 1 and target_features.size(0) > 1:
        s_centered = source_features - s_mean.unsqueeze(0)
        t_centered = target_features - t_mean.unsqueeze(0)
        s_cov = s_centered.t().mm(s_centered) / (source_features.size(0) - 1)
        t_cov = t_centered.t().mm(t_centered) / (target_features.size(0) - 1)
        loss = loss + (s_cov - t_cov).pow(2).sum()

    return loss


# ===================== 均衡推理 =====================

def balanced_predict(logits, num_classes=10):
    """基于置信度的均衡分配：每类恰好 N/10 个样本"""
    probs = torch.softmax(logits, dim=1)
    n = probs.size(0)
    per_class = n // num_classes
    predictions = torch.full((n,), -1, dtype=torch.long)
    quota = torch.full((num_classes,), per_class, dtype=torch.long)

    flat_probs = probs.flatten()
    sorted_indices = flat_probs.argsort(descending=True)

    for idx in sorted_indices:
        sample_id = idx.item() // num_classes
        class_id = idx.item() % num_classes
        if predictions[sample_id].item() == -1 and quota[class_id] > 0:
            predictions[sample_id] = class_id
            quota[class_id] -= 1

    return predictions


# ===================== 训练循环 =====================

def train_epoch_msda(source_loader, target_loader, models, optimizers, criterions, scaler, mm_weight, global_step, total_steps, lr_schedulers):
    feature_extractor, label_predictor, domain_classifier = models
    optimizer_F, optimizer_C, optimizer_D = optimizers
    class_criterion, domain_criterion = criterions

    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)):
        lamb = adaptive_lambda(global_step + i, total_steps)
        source_data = source_data.npu()
        source_label = source_label.npu()
        target_data = target_data.npu()
        n_source = source_data.shape[0]

        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros(mixed_data.shape[0], 1).npu()
        domain_label[:n_source] = 1

        # Step 1: Domain Classifier
        with torch.autocast("npu"):
            feature = feature_extractor(mixed_data)
            domain_logits = domain_classifier(feature.detach())
            loss_D = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss_D.item()
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Step 2: Feature Extractor + Label Predictor + 矩匹配
        with torch.autocast("npu"):
            class_logits = label_predictor(feature[:n_source])
            domain_logits = domain_classifier(feature)
            loss_adv = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
            loss_mm = moment_matching_loss(feature[:n_source], feature[n_source:])
            loss_F = loss_adv + mm_weight * loss_mm
        running_F_loss += loss_F.item()
        scaler.scale(loss_F).backward()
        scaler.step(optimizer_F)
        scaler.step(optimizer_C)
        scaler.update()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += n_source

        for sched in lr_schedulers:
            sched.step()

        print(i, end='\r')

    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num, global_step + i + 1


# ===================== 伪标签 =====================

def generate_pseudo_labels(feature_extractor, label_predictor, test_loader, threshold):
    """为目标域生成高置信度伪标签"""
    feature_extractor.eval()
    label_predictor.eval()

    all_probs = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.npu()
            logits = label_predictor(feature_extractor(data))
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    max_probs, pseudo_labels = all_probs.max(dim=1)
    mask = max_probs >= threshold

    feature_extractor.train()
    label_predictor.train()
    return pseudo_labels, mask


def train_epoch_pseudo(pseudo_loader, models, optimizers, criterion, scaler):
    feature_extractor, label_predictor = models[0], models[1]
    optimizer_F, optimizer_C = optimizers[0], optimizers[1]

    running_loss, total_hit, total_num = 0.0, 0.0, 0.0
    for i, (data, label) in enumerate(pseudo_loader):
        data = data.npu()
        label = label.npu()

        with torch.autocast("npu"):
            feature = feature_extractor(data)
            logits = label_predictor(feature)
            loss = criterion(logits, label)

        running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer_F)
        scaler.step(optimizer_C)
        scaler.update()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == label).item()
        total_num += data.shape[0]

    return running_loss / (i + 1), total_hit / total_num


# ===================== 推理 =====================

def run_inference(feature_extractor, label_predictor, test_loader):
    feature_extractor.load_state_dict(torch.load(EXTRACTOR_PATH, map_location='npu:0'))
    label_predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location='npu:0'))
    feature_extractor.eval()
    label_predictor.eval()

    all_logits = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.npu()
            logits = label_predictor(feature_extractor(data))
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    predictions = balanced_predict(all_logits)

    df = pd.DataFrame({'id': np.arange(len(predictions)), 'label': predictions.numpy()})
    df.to_csv('DaNN_submission.csv', index=False)
    print(f"均衡推理完成，结果已保存到 DaNN_submission.csv ({len(predictions)} 条)")


# ===================== 保存/加载 =====================

def save_models(feature_extractor, label_predictor):
    torch.save(feature_extractor.state_dict(), EXTRACTOR_PATH)
    torch.save(label_predictor.state_dict(), PREDICTOR_PATH)
    print(f"\n模型已保存: {EXTRACTOR_PATH}, {PREDICTOR_PATH}")


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description='HW11 MSDA 多源域对抗训练')
    parser.add_argument('--mode', choices=['train', 'pseudo', 'infer', 'all'], default='all',
                        help='train=MSDA训练, pseudo=伪标签微调, infer=推理, all=全流程 (默认)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--pseudo-epochs', type=int, default=PSEUDO_EPOCHS)
    parser.add_argument('--pseudo-threshold', type=float, default=PSEUDO_THRESHOLD)
    parser.add_argument('--lamb-max', type=float, default=LAMB_MAX)
    parser.add_argument('--mm-weight', type=float, default=MM_WEIGHT)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()

    # ---- 数据加载 ----
    source_datasets = []
    for domain in SOURCE_DOMAINS:
        path = f'{MSDA_DATA_DIR}/{domain}'
        try:
            ds = ImageFolder(path, transform=source_transform)
            source_datasets.append(ds)
            print(f"  源域 {domain}: {len(ds)} 张")
        except FileNotFoundError:
            print(f"  源域 {domain}: 未找到 {path}，请先运行 hw11_preprocess.py")
            return

    multi_source = MultiSourceDataset(source_datasets)
    source_loader = DataLoader(multi_source, batch_size=32, shuffle=True, num_workers=2)

    target_dataset = ImageFolder(TARGET_DIR, transform=target_transform)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(target_dataset, batch_size=128, shuffle=False, num_workers=2)

    # ---- 模型初始化 ----
    feature_extractor = FeatureExtractor().npu()
    label_predictor = LabelPredictor().npu()
    domain_classifier = DomainClassifier().npu()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=args.lr)
    optimizer_C = optim.Adam(label_predictor.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(domain_classifier.parameters(), lr=args.lr)

    scaler = torch.amp.GradScaler("npu")

    # LR 调度: warmup + cosine，按 step 调度
    steps_per_epoch = len(source_loader)
    total_steps = args.epochs * steps_per_epoch
    sched_F = create_lr_scheduler(optimizer_F, total_steps)
    sched_C = create_lr_scheduler(optimizer_C, total_steps)
    sched_D = create_lr_scheduler(optimizer_D, total_steps)
    lr_schedulers = [sched_F, sched_C, sched_D]

    models = (feature_extractor, label_predictor, domain_classifier)
    optimizers = (optimizer_F, optimizer_C, optimizer_D)
    criterions = (class_criterion, domain_criterion)

    # ---- Phase 1: MSDA 训练 ----
    if args.mode in ('train', 'all'):
        print(f"\n=== MSDA 训练 ({args.epochs} epochs) ===")
        print(f"  lambda: 0 → {args.lamb_max} (自适应), mm_weight={args.mm_weight}")
        print(f"  lr: {args.lr}, warmup={WARMUP_RATIO}, total_steps={total_steps}")
        print(f"  源域数: {len(SOURCE_DOMAINS)}, 总样本: {len(multi_source)}")
        global_step = 0
        try:
            for epoch in range(args.epochs):
                d_loss, f_loss, acc, global_step = train_epoch_msda(
                    source_loader, target_loader, models, optimizers, criterions,
                    scaler, args.mm_weight, global_step, total_steps, lr_schedulers,
                )
                save_models(feature_extractor, label_predictor)
                current_lamb = adaptive_lambda(global_step, total_steps)
                current_lr = optimizer_F.param_groups[0]['lr']
                print(f'epoch {epoch:>3d}: D loss: {d_loss:6.4f}, '
                      f'F loss: {f_loss:6.4f}, acc {acc:6.4f}, '
                      f'λ={current_lamb:.3f}, lr={current_lr:.5f}')
        except KeyboardInterrupt:
            print("\n训练中断，保存当前模型...")
            save_models(feature_extractor, label_predictor)

    # ---- Phase 2: 伪标签半监督 ----
    if args.mode in ('pseudo', 'all'):
        print(f"\n=== 伪标签半监督训练 ({args.pseudo_epochs} epochs, 阈值={args.pseudo_threshold}) ===")
        pseudo_labels, mask = generate_pseudo_labels(
            feature_extractor, label_predictor, test_loader, args.pseudo_threshold
        )
        n_pseudo = mask.sum().item()
        print(f"  高置信度伪标签: {n_pseudo}/{len(mask)} 个 ({n_pseudo / len(mask) * 100:.1f}%)")

        if n_pseudo > 0:
            target_dataset_no_aug = ImageFolder(TARGET_DIR, transform=test_transform)
            pseudo_indices = mask.nonzero(as_tuple=True)[0].tolist()
            pseudo_subset = torch.utils.data.Subset(target_dataset_no_aug, pseudo_indices)
            pseudo_labels_subset = pseudo_labels[mask]
            pseudo_ds = torch.utils.data.TensorDataset(
                torch.stack([target_dataset_no_aug[i][0] for i in pseudo_indices]),
                pseudo_labels_subset,
            )
            pseudo_loader = DataLoader(pseudo_ds, batch_size=32, shuffle=True)

            try:
                for epoch in range(args.pseudo_epochs):
                    loss, acc = train_epoch_pseudo(
                        pseudo_loader, models, optimizers, class_criterion, scaler,
                    )
                    save_models(feature_extractor, label_predictor)
                    print(f'pseudo epoch {epoch:>3d}: loss {loss:6.4f}, acc {acc:6.4f}')
            except KeyboardInterrupt:
                print("\n伪标签训练中断，保存当前模型...")
                save_models(feature_extractor, label_predictor)
        else:
            print("  无高置信度伪标签，跳过")

    # ---- Phase 3: 均衡推理 ----
    if args.mode in ('infer', 'all'):
        print("\n=== 均衡推理 ===")
        run_inference(feature_extractor, label_predictor, test_loader)


if __name__ == '__main__':
    main()
