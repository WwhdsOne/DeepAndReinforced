"""
HW06 - 动漫人脸生成 (WGAN-GP + DCGAN)
基于李宏毅老师 ML2021 课程作业，适配昇腾 NPU 训练。
"""

import os
import glob
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torch_npu


# ============================================================
#  参数配置
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="HW06 GAN NPU 训练")
    parser.add_argument("--data_dir", type=str, default="./faces", help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64, help="模型通道基数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epoch", type=int, default=300)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--bf16", type=bool, default=True, help="使用 bfloat16 混合精度训练")
    parser.add_argument("--mode", type=str, default="both", choices=["train", "generate", "both"],
                        help="运行模式: train=仅训练, generate=仅推理, both=训练+推理")
    return parser.parse_args()


# ============================================================
#  工具函数
# ============================================================
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
#  数据集
# ============================================================
class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, "*"))
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return CrypkoDataset(fnames, transform)


# ============================================================
#  模型
# ============================================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Input:  (N, in_dim)
    Output: (N, 3, 64, 64)
    """

    def __init__(self, in_dim, dim=64):
        super().__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False
                ),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
            )

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU(),
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        return self.l2_5(y)


class Discriminator(nn.Module):
    """
    Input:  (N, 3, 64, 64)
    Output: (N, )
    """

    def __init__(self, in_dim, dim=64):
        super().__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.LeakyReLU(0.2),
            )

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        return y.view(-1)


# ============================================================
#  WGAN-GP 梯度惩罚
# ============================================================
def gradient_penalty(D, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ============================================================
#  训练
# ============================================================
def train(args):
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"设备: {device}, bf16: {'启用' if args.bf16 else '禁用'}")

    same_seeds(args.seed)

    # 目录
    log_dir = os.path.join(args.output_dir, "logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 数据
    dataset = get_dataset(args.data_dir)
    print(f"数据集样本数: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 模型
    G = Generator(in_dim=args.z_dim, dim=args.dim).to(device)
    D = Discriminator(in_dim=3, dim=args.dim).to(device)
    G.train()
    D.train()

    # 优化器 (WGAN-GP 论文推荐 betas)
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))

    # 固定采样噪声（用于可视化）
    z_sample = torch.randn(100, args.z_dim, device=device)

    # 断点续训
    start_epoch = 0
    steps = 0
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "G" in ckpt:
            G.load_state_dict(ckpt["G"])
            D.load_state_dict(ckpt["D"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
            start_epoch = ckpt["epoch"]
            steps = ckpt["steps"]
            print(f"从检查点恢复: Epoch {start_epoch}, Steps {steps}")

    # 损失记录文件
    loss_log_path = os.path.join(args.output_dir, "loss_log.txt")
    if start_epoch == 0:
        with open(loss_log_path, "w") as f:
            f.write("epoch,loss_d,loss_g\n")

    # 训练循环
    for e in range(start_epoch, args.n_epoch):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        n_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {e+1}/{args.n_epoch}", leave=False)
        for batch_idx, data in enumerate(progress_bar):
            imgs = data.to(device, non_blocking=True)
            bs = imgs.size(0)

            # --- 训练 D ---
            z = torch.randn(bs, args.z_dim, device=device)
            opt_D.zero_grad()

            if args.bf16:
                with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                    f_imgs = G(z)
                    d_real = D(imgs)
                    d_fake = D(f_imgs.detach())
                # 梯度惩罚保持 float32 以保证数值稳定
                gp = gradient_penalty(D.float(), imgs, f_imgs.detach().float(), device)
                loss_D = -torch.mean(d_real) + torch.mean(d_fake) + args.lambda_gp * gp
            else:
                f_imgs = G(z)
                loss_D = -torch.mean(D(imgs)) + torch.mean(D(f_imgs.detach()))
                gp = gradient_penalty(D, imgs, f_imgs.detach(), device)
                loss_D = loss_D + args.lambda_gp * gp

            loss_D.backward()
            opt_D.step()

            # --- 训练 G (每 n_critic 步) ---
            loss_g_val = 0.0
            if steps % args.n_critic == 0:
                z = torch.randn(bs, args.z_dim, device=device)
                opt_G.zero_grad()

                if args.bf16:
                    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                        f_imgs = G(z)
                        loss_G = -torch.mean(D(f_imgs))
                else:
                    f_imgs = G(z)
                    loss_G = -torch.mean(D(f_imgs))

                loss_G.backward()
                opt_G.step()
                loss_g_val = loss_G.item()

            epoch_loss_d += loss_D.item()
            epoch_loss_g += loss_g_val
            steps += 1
            n_batches += 1

            progress_bar.set_postfix({
                'Loss_D': f"{loss_D.item():.4f}",
                'Loss_G': f"{loss_g_val:.4f}",
                'Step': steps,
            })

        avg_d = epoch_loss_d / n_batches
        avg_g = epoch_loss_g / n_batches
        print(f"Epoch {e+1} 平均 Loss_D: {avg_d:.4f}, Loss_G: {avg_g:.4f}")

        # 每个 epoch 的 loss 追加到文件
        with open(loss_log_path, "a") as f:
            f.write(f"{e+1},{avg_d:.4f},{avg_g:.4f}\n")

        # 每 10 epoch 保存生成样本
        if (e + 1) % 10 == 0 or e == 0:
            G.eval()
            with torch.no_grad():
                if args.bf16:
                    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                        f_imgs_sample = (G(z_sample) + 1) / 2.0
                else:
                    f_imgs_sample = (G(z_sample) + 1) / 2.0
            filename = os.path.join(log_dir, f"Epoch_{e+1:03d}.jpg")
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            print(f"  保存样本到 {filename}")
            G.train()

        # 每 5 epoch 保存检查点
        if (e + 1) % 5 == 0 or e == 0:
            torch.save(
                {
                    "epoch": e + 1,
                    "steps": steps,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                },
                os.path.join(ckpt_dir, "checkpoint.pth"),
            )
            torch.save(G.state_dict(), os.path.join(ckpt_dir, "G.pth"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, "D.pth"))
            print(f"  检查点已保存")

    print("训练完成！")


# ============================================================
#  推理
# ============================================================
def generate(args, n_output=1000):
    device = torch.device("npu" if torch.npu.is_available() else "cpu")

    G = Generator(in_dim=args.z_dim, dim=args.dim).to(device)
    ckpt_path = os.path.join(args.output_dir, "checkpoints", "G.pth")
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()

    z_sample = torch.randn(n_output, args.z_dim, device=device)
    with torch.no_grad():
        if args.bf16:
            with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                imgs_sample = (G(z_sample) + 1) / 2.0
        else:
            imgs_sample = (G(z_sample) + 1) / 2.0

    result_dir = os.path.join(args.output_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    filename = os.path.join(result_dir, "result.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    print(f"生成结果已保存到 {filename}")

    # 逐张保存
    for i in range(n_output):
        torchvision.utils.save_image(
            imgs_sample[i], os.path.join(result_dir, f"{i+1}.jpg")
        )
    print(f"已保存 {n_output} 张图像到 {result_dir}")


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    print(f"参数: {vars(args)}")
    if args.mode in ("train", "both"):
        train(args)
    if args.mode in ("generate", "both"):
        generate(args)
