"""
HW11 DaNN 推理脚本 - 仅加载已训练的 .bin 权重生成提交 CSV
用法: uv run python hw11_infer_npu.py
"""

import numpy as np
import cv2
import torch
import torch_npu
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd


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


target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

target_dataset = ImageFolder('../real_or_drawing/test_data', transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

feature_extractor = FeatureExtractor().npu()
label_predictor = LabelPredictor().npu()

feature_extractor.load_state_dict(torch.load('extractor_model.bin', map_location='npu:0'))
label_predictor.load_state_dict(torch.load('predictor_model.bin', map_location='npu:0'))
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
