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

# ===================== 初始化 =====================

feature_extractor = FeatureExtractor().npu()
label_predictor = LabelPredictor().npu()
domain_classifier = DomainClassifier().npu()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

scaler = torch.amp.GradScaler("npu")

LAMB = 0.1

# ===================== 训练 =====================

def train_epoch(source_dataloader, target_dataloader, lamb):
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


if __name__ == '__main__':
    for epoch in range(200):
        train_D_loss, train_F_loss, train_acc = train_epoch(
            source_dataloader, target_dataloader, lamb=LAMB
        )
        torch.save(feature_extractor.state_dict(), 'extractor_model.bin')
        torch.save(label_predictor.state_dict(), 'predictor_model.bin')
        print(f'epoch {epoch:>3d}: train D loss: {train_D_loss:6.4f}, '
              f'train F loss: {train_F_loss:6.4f}, acc {train_acc:6.4f}')

    # ===================== 推理 =====================

    result = []
    label_predictor.eval()
    feature_extractor.eval()
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.npu()
        class_logits = label_predictor(feature_extractor(test_data))
        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)

    result = np.concatenate(result)
    df = pd.DataFrame({'id': np.arange(len(result)), 'label': result})
    df.to_csv('DaNN_submission.csv', index=False)
    print(f"推理完成，结果已保存到 DaNN_submission.csv ({len(result)} 条)")
