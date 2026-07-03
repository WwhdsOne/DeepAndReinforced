"""
HW08 异常检测 - 升腾 NPU 专用版
Mahalanobis + Feature Distance + 多模型 Ensemble + 多尺度 CNN + SSIM Loss

用法:
    python train_hw08_npu.py --data_path /path/to/data --epochs 500
    python train_hw08_npu.py --data_path /path/to/data --ensemble
    python train_hw08_npu.py --infer_only --data_path /path/to/data
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torchvision.transforms as transforms

import torch_npu
from torch_npu.contrib import transfer_to_npu

DEVICE = torch.device('npu')


# ============================================================
# 随机种子
# ============================================================
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


# ============================================================
# 数据集
# ============================================================
class CustomTensorDataset(TensorDataset):
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),
            transforms.Lambda(lambda x: 2.0 * x / 255.0 - 1.0),
        ])

    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.tensors)


# ============================================================
# 模型定义
# ============================================================
class fcn_autoencoder(nn.Module):
    """宽全连接自编码器"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 4096), nn.BatchNorm1d(4096), nn.PReLU(),
            nn.Linear(4096, 2048), nn.BatchNorm1d(2048), nn.PReLU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.PReLU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.PReLU(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.PReLU(),
            nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.PReLU(),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.PReLU(),
            nn.Linear(2048, 4096), nn.BatchNorm1d(4096), nn.PReLU(),
            nn.Linear(4096, 64 * 64 * 3), nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class conv_autoencoder(nn.Module):
    """卷积自编码器 - 重写版: 32→64→128→256, bottleneck=128d, 多尺度输出"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.PReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 128),
        )
        self.unbottleneck = nn.Sequential(
            nn.Linear(128, 256 * 4 * 4), nn.Unflatten(1, (256, 4, 4)),
        )
        self.dec_4to8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.dec_8to16 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.PReLU(),
        )
        self.dec_16to32 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.PReLU(),
        )
        self.dec_32to64 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Tanh(),
        )
        self.proj_16 = nn.Conv2d(64, 3, 1)
        self.proj_32 = nn.Conv2d(32, 3, 1)

    def encode(self, x):
        return self.bottleneck(self.encoder(x))

    def forward(self, x):
        h = self.encoder(x)
        z = self.bottleneck(h)
        h = self.unbottleneck(z)
        f8 = self.dec_4to8(h)
        f16 = self.dec_8to16(f8)
        f32 = self.dec_16to32(f16)
        out = self.dec_32to64(f32)
        return out, self.proj_16(f16), self.proj_32(f32)


def ssim_loss(x, y, window_size=7):
    """简化版 1-SSIM loss"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim.mean()


# ============================================================
# Warmup + Cosine Annealing
# ============================================================
def warmup_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 单模型训练
# ============================================================
def train_model(model, model_name, train_dataloader, amp_dtype, scaler, args):
    print(f"\n{'='*50}\n训练模型: {model_name}\n{'='*50}")

    if model_name == 'cnn':
        criterion = nn.L1Loss()
        optimizer = AdamW(model.parameters(), lr=args.cnn_lr)
    else:
        criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=args.lr)

    warmup_steps = args.warmup_steps
    total_steps = args.epochs * len(train_dataloader)
    scheduler = warmup_lr_scheduler(optimizer, warmup_steps, total_steps)

    best_loss = np.inf
    global_step = 0

    for epoch in range(args.epochs):
        losses = []
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        for data in pbar:
            global_step += 1
            img = data.float().npu()

            if model_name == 'fcn':
                inp = img.view(img.shape[0], -1)
            else:
                inp = img

            with torch.amp.autocast(device_type='npu', dtype=amp_dtype):
                output = model(inp)
                if model_name == 'cnn':
                    out_64, out_16, out_32 = output
                    target_16 = F.interpolate(img, size=16, mode='bilinear', align_corners=False)
                    target_32 = F.interpolate(img, size=32, mode='bilinear', align_corners=False)
                    loss = criterion(out_64, img) + ssim_loss(out_64, img) \
                         + 0.5 * criterion(out_16, target_16) + 0.5 * criterion(out_32, target_32)
                else:
                    loss = criterion(output, inp)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{scheduler.get_last_lr()[0]:.6f}')

        mean_loss = np.mean(losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, os.path.join(args.output_dir, f'best_model_{model_name}.pt'))
        torch.save(model, os.path.join(args.output_dir, f'last_model_{model_name}.pt'))
        print(f"Epoch {epoch+1}/{args.epochs} | loss={mean_loss:.6f} | best={best_loss:.6f}")

    print(f"{model_name} 训练完成! 最优损失: {best_loss:.6f}")
    return model


# ============================================================
# Latent 高斯拟合
# ============================================================
def fit_latent_gaussian(model, model_name, train_dataloader):
    print(f"\n拟合 {model_name} latent 高斯分布...")
    model.eval()
    latents = []
    with torch.no_grad():
        for data in tqdm(train_dataloader, desc='提取 latent'):
            img = data.float().npu()
            inp = img.view(img.shape[0], -1) if model_name == 'fcn' else img
            z = model.encode(inp)
            latents.append(z.cpu())

    latents = torch.cat(latents, dim=0)
    mean = latents.mean(dim=0)
    cov = torch.cov(latents.T) + torch.eye(latents.shape[1]) * 1e-5
    cov_inv = torch.linalg.inv(cov)
    print(f"Latent 维度: {latents.shape[1]}, 样本数: {latents.shape[0]}")
    return mean.npu(), cov_inv.npu(), latents


# ============================================================
# 推理评分
# ============================================================
def score_model(model, model_name, mean, cov_inv, train_latents, test_dataloader):
    eval_loss = nn.MSELoss(reduction='none')
    model.eval()

    recon_scores, maha_scores, feat_dists = [], [], []

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc=f'推理({model_name})'):
            img = data.float().npu()
            inp = img.view(img.shape[0], -1) if model_name == 'fcn' else img

            output = model(inp)
            if model_name == 'cnn':
                out = output[0]
                recon = eval_loss(out, img).sum([1, 2, 3])
            else:
                recon = eval_loss(output, inp).sum(-1)
            recon_scores.append(torch.sqrt(recon))

            z = model.encode(inp)
            diff = z - mean
            maha = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
            maha_scores.append(maha)

            for zi in z:
                dists = torch.cdist(zi.unsqueeze(0), train_latents.npu()).squeeze(0)
                feat_dists.append(dists.min().item())

    recon_scores = torch.cat(recon_scores).cpu().numpy()
    maha_scores = torch.cat(maha_scores).cpu().numpy()
    feat_dists = np.array(feat_dists)

    print(f"  重建误差: min={recon_scores.min():.4f}, max={recon_scores.max():.4f}, mean={recon_scores.mean():.4f}")
    print(f"  Mahalanobis: min={maha_scores.min():.4f}, max={maha_scores.max():.4f}, mean={maha_scores.mean():.4f}")
    print(f"  Feature Dist: min={feat_dists.min():.4f}, max={feat_dists.max():.4f}, mean={feat_dists.mean():.4f}")
    return recon_scores, maha_scores, feat_dists


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='HW08 异常检测 - 升腾 NPU 专用版')
    parser.add_argument('--data_path', type=str, default='data-bin')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cnn_lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=19530615)
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--maha_weight', type=float, default=0.3)
    parser.add_argument('--feat_weight', type=float, default=0.1)
    parser.add_argument('--infer_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    same_seeds(args.seed)

    print(f"设备: NPU ({torch.npu.get_device_name(0)})")

    # 混合精度
    if args.fp8:
        try:
            dummy = torch.randn(2, 2, device=DEVICE)
            with torch.amp.autocast(device_type='npu', dtype=torch.float8_e4m3fn):
                _ = dummy + dummy
            amp_dtype = torch.float8_e4m3fn
            print("[fp8] 使用 float8_e4m3fn 混合精度训练")
        except Exception:
            amp_dtype = torch.bfloat16
            print("[fp8] 当前 NPU 不支持 fp8 autocast，回退到 bf16")
    else:
        amp_dtype = torch.float16
        print("[fp16] 使用 float16 混合精度训练")

    scaler = torch.amp.GradScaler('npu') if amp_dtype == torch.float16 else None

    # 加载数据
    train_data = np.load(os.path.join(args.data_path, 'trainingset.npy'), allow_pickle=True)
    test_data = np.load(os.path.join(args.data_path, 'testingset.npy'), allow_pickle=True)
    print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")

    train_dataset = CustomTensorDataset(torch.from_numpy(train_data))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)

    test_dataset = CustomTensorDataset(torch.tensor(test_data, dtype=torch.float32))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)

    model_names = ['fcn']
    if args.ensemble:
        model_names.append('cnn')

    if args.infer_only:
        models = {}
        gaussians = {}
        for name in model_names:
            ckpt = args.checkpoint or os.path.join(args.output_dir, f'last_model_{name}.pt')
            models[name] = torch.load(ckpt, weights_only=False).to(DEVICE)
        for name in model_names:
            mean, cov_inv, latents = fit_latent_gaussian(models[name], name, train_dataloader)
            gaussians[name] = (mean, cov_inv, latents)
    else:
        models = {}
        for name in model_names:
            if name == 'fcn':
                models[name] = fcn_autoencoder().to(DEVICE)
            elif name == 'cnn':
                models[name] = conv_autoencoder().to(DEVICE)
            train_model(models[name], name, train_dataloader, amp_dtype, scaler, args)

        gaussians = {}
        for name in model_names:
            mean, cov_inv, latents = fit_latent_gaussian(models[name], name, train_dataloader)
            gaussians[name] = (mean, cov_inv, latents)
            torch.save({'mean': mean.cpu(), 'cov_inv': cov_inv.cpu()},
                       os.path.join(args.output_dir, f'gaussian_{name}.pt'))

    # 推理评分
    print(f"\n{'='*50}\n推理评分\n{'='*50}")

    all_recon, all_maha, all_feat = [], [], []
    for name in model_names:
        recon, maha, feat = score_model(models[name], name, *gaussians[name], test_dataloader)
        all_recon.append(recon)
        all_maha.append(maha)
        all_feat.append(feat)

    recon_avg = np.mean(all_recon, axis=0)
    maha_avg = np.mean(all_maha, axis=0)
    feat_avg = np.mean(all_feat, axis=0)

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    combined = normalize(recon_avg) + args.maha_weight * normalize(maha_avg) + args.feat_weight * normalize(feat_avg)

    print(f"\n组合评分 (recon + {args.maha_weight}×maha + {args.feat_weight}×feat):")
    print(f"  min={combined.min():.4f}, max={combined.max():.4f}, mean={combined.mean():.4f}")

    out_file = os.path.join(args.output_dir, 'PREDICTION_FILE.csv')
    df = pd.DataFrame(combined.reshape(-1, 1), columns=['score'])
    df.to_csv(out_file, index_label='Id')
    print(f"预测已保存到: {out_file}")


if __name__ == '__main__':
    main()
