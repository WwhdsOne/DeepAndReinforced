"""
HW11 数据预处理：生成 4 种源域图像
Sobel / Canny / Laplacian / Style Transfer (AdaIN)

用法:
  python hw11_preprocess.py                                    # 默认用 OpenCV 素描（快）
  python hw11_preprocess.py --style-mode adain                 # 真正的 AdaIN 风格迁移
  python hw11_preprocess.py --style-mode adain --style-img 0/0.bmp  # 指定风格参考图
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# ===================== 边缘检测变换 =====================

def sobel_transform(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = np.random.choice([3, 5])
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.clip(edges, 0, 255).astype(np.uint8)


def canny_transform(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    low = np.random.randint(100, 200)
    high = np.random.randint(low + 30, min(low + 150, 350))
    return cv2.Canny(gray, low, high)


def laplacian_transform(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = np.random.choice([1, 3, 5])
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


def sketch_transform(img_bgr):
    """铅笔素描效果 — 快速近似（无需预训练模型）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur_k = np.random.choice([15, 21, 25])
    blur = cv2.GaussianBlur(inv, (blur_k, blur_k), sigmaX=0, sigmaY=0)
    return cv2.divide(gray, 255 - blur, scale=256)


# ===================== AdaIN 风格迁移 =====================

ADAIN_DECODER_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
)
ADAIN_DECODER_CACHE = "decoder.pth"


def _build_decoder():
    """与 naoto0804/pytorch-AdaIN 完全一致的 decoder 架构"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(64, 3, 3, 1, 1),
    )


def _download_decoder():
    if os.path.exists(ADAIN_DECODER_CACHE):
        return ADAIN_DECODER_CACHE
    print(f"下载 AdaIN decoder 权重: {ADAIN_DECODER_URL}")
    try:
        import urllib.request
        urllib.request.urlretrieve(ADAIN_DECODER_URL, ADAIN_DECODER_CACHE)
        print("下载完成")
        return ADAIN_DECODER_CACHE
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def _adain_normalize(content_feat, style_feat):
    """Adaptive Instance Normalization"""
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True)
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True)
    return (content_feat - c_mean) / (c_std + 1e-5) * s_std + s_mean


class AdaINStyleTransfer:
    """预训练 VGG encoder + AdaIN decoder 风格迁移"""

    def __init__(self, device='cpu'):
        import torch
        from torchvision.models import vgg19
        from torchvision import transforms as T

        self.device = device
        self.torch = torch

        # VGG encoder: 取到 relu4_1 (features[:21])
        vgg = vgg19(pretrained=True).features[:21].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.encoder = vgg.to(device)

        # Decoder
        decoder_path = _download_decoder()
        self.decoder = _build_decoder()
        if decoder_path:
            state_dict = torch.load(decoder_path, map_location=device)
            self.decoder.load_state_dict(state_dict)
            self.available = True
        else:
            self.available = False
        self.decoder = self.decoder.to(device).eval()

        # ImageNet 归一化
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = T.Compose([T.ToTensor(), self.normalize])
        self.inv_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                         std=[1/0.229, 1/0.224, 1/0.225])

    def transfer(self, content_img_bgr, style_img_bgr, alpha=0.8):
        """
        将 style_img 的风格迁移到 content_img 上
        输入: BGR numpy array (OpenCV 格式)
        输出: 灰度 numpy array (32x32)
        """
        import torch
        from PIL import Image

        if not self.available:
            return sketch_transform(content_img_bgr)

        # 转 PIL → Tensor
        content_rgb = cv2.cvtColor(content_img_bgr, cv2.COLOR_BGR2RGB)
        style_rgb = cv2.cvtColor(style_img_bgr, cv2.COLOR_BGR2RGB)
        content_pil = Image.fromarray(content_rgb).resize((32, 32))
        style_pil = Image.fromarray(style_rgb).resize((32, 32))

        content_t = self.to_tensor(content_pil).unsqueeze(0).to(self.device)
        style_t = self.to_tensor(style_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            content_feat = self.encoder(content_t)
            style_feat = self.encoder(style_t)
            stylized_feat = _adain_normalize(content_feat, style_feat)
            stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
            output = self.decoder(stylized_feat)

        # Tensor → numpy 灰度
        output = output.squeeze(0).cpu().clamp(0, 1)
        output_np = output.permute(1, 2, 0).numpy()
        gray = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return gray


# ===================== 主函数 =====================

TRANSFORMS = {
    'sobel': sobel_transform,
    'canny': canny_transform,
    'laplacian': laplacian_transform,
    'sketch': sketch_transform,
}


def main():
    parser = argparse.ArgumentParser(description='HW11 多源域数据预处理')
    parser.add_argument('--input', default='../real_or_drawing/train_data')
    parser.add_argument('--output', default='../real_or_drawing/msda_data')
    parser.add_argument('--style-mode', choices=['fast', 'adain'], default='fast',
                        help='fast=OpenCV 素描 (默认), adain=预训练 AdaIN 风格迁移')
    parser.add_argument('--style-img', default=None,
                        help='风格参考图路径 (相对于 input 目录, 默认随机选一张目标域图)')
    parser.add_argument('--device', default='cpu', help='推理设备 (cpu/npu:0)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # 选择风格参考图（用于 AdaIN）
    style_reference = None
    if args.style_mode == 'adain':
        if args.style_img:
            style_path = input_dir / args.style_img
        else:
            # 默认选第一类第一张图作为风格参考
            first_cls = sorted(os.listdir(input_dir))[0]
            first_img = sorted(os.listdir(input_dir / first_cls))[0]
            style_path = input_dir / first_cls / first_img
        style_reference = cv2.imread(str(style_path))
        print(f"风格参考图: {style_path}")

    transforms_map = dict(TRANSFORMS)
    adain_transfer = None
    if args.style_mode == 'adain':
        print("初始化 AdaIN 风格迁移模型...")
        adain_transfer = AdaINStyleTransfer(device=args.device)
        if not adain_transfer.available:
            print("AdaIN decoder 权重不可用，回退到 OpenCV 素描")
        else:
            def adain_transform(img_bgr):
                return adain_transfer.transfer(img_bgr, style_reference)
            transforms_map['sketch'] = adain_transform
            print("使用 AdaIN 风格迁移")

    classes = sorted([d for d in os.listdir(input_dir) if (input_dir / d).is_dir()])
    total = sum(len(list((input_dir / c).iterdir())) for c in classes)

    for domain_name, transform_fn in transforms_map.items():
        domain_dir = output_dir / domain_name
        for cls in classes:
            (domain_dir / cls).mkdir(parents=True, exist_ok=True)

        img_files = []
        for cls in classes:
            img_files.extend([(cls, f) for f in sorted(os.listdir(input_dir / cls))])

        processed = 0
        for cls, filename in img_files:
            img_path = input_dir / cls / filename
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            result = transform_fn(img_bgr)
            out_path = domain_dir / cls / filename
            cv2.imwrite(str(out_path), result)
            processed += 1
            print(f'[{domain_name}] {processed}/{total} {cls}/{filename}', end='\r')
        print()

    print(f"\n预处理完成，输出目录: {output_dir}")
    for domain_name in transforms_map:
        count = sum(len(list((output_dir / domain_name / c).iterdir())) for c in classes)
        print(f"  {domain_name}: {count} 张")


if __name__ == '__main__':
    main()
