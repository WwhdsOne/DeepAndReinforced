"""
HW11 - Domain Adversarial Training (DaNN) - NPU 版本
从 HW11_NPU.ipynb 转换而来，lambda=0.1
"""

import numpy as np
import cv2
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd

# ===================== 数据变换 =====================

source_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
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

# ===================== 数据加载 =====================

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

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

LAMB = 0.1
NUM_EPOCHS = 200
EXTRACTOR_PATH = 'extractor_model.bin'
PREDICTOR_PATH = 'predictor_model.bin'

# ===================== 训练 =====================

def train_epoch(source_dataloader, target_dataloader, lamb,
                feature_extractor, label_predictor, domain_classifier,
                domain_criterion, class_criterion,
                optimizer_F, optimizer_C, optimizer_D, scaler):
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.npu()
        source_label = source_label.npu()
        target_data = target_data.npu()

        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).npu()
        domain_label[:source_data.shape[0]] = 1

        # Step 1: 训练 Domain Classifier
        with torch.autocast("npu"):
            feature = feature_extractor(mixed_data)
            domain_logits = domain_classifier(feature.detach())
            loss_D = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss_D.item()
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Step 2: 训练 Feature Extractor + Label Predictor
        with torch.autocast("npu"):
            class_logits = label_predictor(feature[:source_data.shape[0]])
            domain_logits = domain_classifier(feature)
            loss_F = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss_F.item()
        scaler.scale(loss_F).backward()
        scaler.step(optimizer_F)
        scaler.step(optimizer_C)
        scaler.update()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num


def save_models(feature_extractor, label_predictor):
    torch.save(feature_extractor.state_dict(), EXTRACTOR_PATH)
    torch.save(label_predictor.state_dict(), PREDICTOR_PATH)
    print(f"\n模型已保存: {EXTRACTOR_PATH}, {PREDICTOR_PATH}")


def train(args, feature_extractor, label_predictor, domain_classifier,
          class_criterion, domain_criterion,
          optimizer_F, optimizer_C, optimizer_D, scaler,
          source_dataloader, target_dataloader):
    try:
        for epoch in range(args.epochs):
            train_D_loss, train_F_loss, train_acc = train_epoch(
                source_dataloader, target_dataloader, lamb=LAMB,
                feature_extractor=feature_extractor,
                label_predictor=label_predictor,
                domain_classifier=domain_classifier,
                domain_criterion=domain_criterion,
                class_criterion=class_criterion,
                optimizer_F=optimizer_F, optimizer_C=optimizer_C,
                optimizer_D=optimizer_D, scaler=scaler,
            )
            save_models(feature_extractor, label_predictor)
            print(f'epoch {epoch:>3d}: train D loss: {train_D_loss:6.4f}, '
                  f'train F loss: {train_F_loss:6.4f}, acc {train_acc:6.4f}')
    except KeyboardInterrupt:
        print("\n训练中断，正在保存当前模型...")
        save_models(feature_extractor, label_predictor)


def infer(feature_extractor, label_predictor, test_dataloader):
    feature_extractor.load_state_dict(torch.load(EXTRACTOR_PATH, map_location='npu:0'))
    label_predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location='npu:0'))
    feature_extractor.eval()
    label_predictor.eval()

    result = []
    with torch.no_grad():
        for test_data, _ in test_dataloader:
            test_data = test_data.npu()
            class_logits = label_predictor(feature_extractor(test_data))
            x = torch.argmax(class_logits, dim=1).cpu().numpy()
            result.append(x)

    result = np.concatenate(result)
    df = pd.DataFrame({'id': np.arange(len(result)), 'label': result})
    df.to_csv('DaNN_submission.csv', index=False)
    print(f"推理完成，结果已保存到 DaNN_submission.csv ({len(result)} 条)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HW11 DaNN 域对抗训练')
    parser.add_argument('--mode', choices=['train', 'infer', 'all'], default='all',
                        help='train=仅训练, infer=仅推理(加载已保存权重), all=训练+推理 (默认)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help=f'训练轮数 (默认 {NUM_EPOCHS})')
    args = parser.parse_args()

    feature_extractor = FeatureExtractor().npu()
    label_predictor = LabelPredictor().npu()
    domain_classifier = DomainClassifier().npu()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())

    scaler = torch.amp.GradScaler("npu")

    if args.mode in ('train', 'all'):
        train(args, feature_extractor, label_predictor, domain_classifier,
              class_criterion, domain_criterion,
              optimizer_F, optimizer_C, optimizer_D, scaler,
              source_dataloader, target_dataloader)

    if args.mode in ('infer', 'all'):
        infer(feature_extractor, label_predictor, test_dataloader)
