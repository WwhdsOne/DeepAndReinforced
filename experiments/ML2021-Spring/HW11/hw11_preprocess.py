"""
HW11 数据预处理：生成 4 种源域图像
Sobel / Canny / Laplacian / Pencil Sketch

用法: python hw11_preprocess.py --input real_or_drawing/train_data --output real_or_drawing/msda_data
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def sobel_transform(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    return edges


def canny_transform(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 170, 300)


def laplacian_transform(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    result = np.abs(lap)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def sketch_transform(img_bgr):
    """铅笔素描效果 — 快速近似 Style Transfer（无需预训练模型）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch


def sketch_stylized(img_bgr):
    """使用预训练 VGG 的 AdaIN 风格迁移（可选，需下载 decoder 权重）
    若权重不可用则自动回退到 sketch_transform"""
    try:
        import torch
        from torchvision.models import vgg19
        from torchvision import transforms as T

        vgg = vgg19(pretrained=True).features[:21].eval()
        for p in vgg.parameters():
            p.requires_grad = False

        to_tensor = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        content_t = to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0, 0)
        style_img = cv2.divide(gray, 255 - blur, scale=256)
        style_pil = cv2.cvtColor(style_img, cv2.COLOR_GRAY2RGB)
        style_t = to_tensor(style_pil).unsqueeze(0)

        content_feat = vgg(content_t)
        style_feat = vgg(style_t)

        def calc_mean_std(feat):
            b, c = feat.size(0), feat.size(1)
            size = feat.size(2) * feat.size(3)
            f = feat.view(b, c, size)
            mean = f.mean(dim=2, keepdim=True).expand_as(f)
            std = f.std(dim=2, keepdim=True).expand_as(f)
            return mean, std

        c_mean, c_std = calc_mean_std(content_feat)
        s_mean, s_std = calc_mean_std(style_feat)
        stylized = (content_feat - c_mean) / (c_std + 1e-5) * s_std + s_mean
        stylized = stylized.clamp(0, 1)
        out = stylized.squeeze(0).mean(dim=0)
        out = (out * 255).byte().cpu().numpy()
        return out
    except Exception:
        return sketch_transform(img_bgr)


TRANSFORMS = {
    'sobel': sobel_transform,
    'canny': canny_transform,
    'laplacian': laplacian_transform,
    'sketch': sketch_transform,
}


def main():
    parser = argparse.ArgumentParser(description='HW11 多源域数据预处理')
    parser.add_argument('--input', default='real_or_drawing/train_data')
    parser.add_argument('--output', default='real_or_drawing/msda_data')
    parser.add_argument('--style-mode', choices=['fast', 'stylized'], default='fast',
                        help='fast=OpenCV 素描 (默认), stylized=VGG AdaIN 风格迁移')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    transforms_map = dict(TRANSFORMS)
    if args.style_mode == 'stylized':
        transforms_map['sketch'] = sketch_stylized
        print("使用 VGG AdaIN 风格迁移（如权重不可用将自动回退到 OpenCV 素描）")

    classes = sorted([d for d in os.listdir(input_dir) if (input_dir / d).is_dir()])
    total = sum(len(list((input_dir / c).iterdir())) for c in classes)
    processed = 0

    for domain_name, transform_fn in transforms_map.items():
        domain_dir = output_dir / domain_name
        for cls in classes:
            (domain_dir / cls).mkdir(parents=True, exist_ok=True)

        img_files = []
        for cls in classes:
            img_files.extend([(cls, f) for f in sorted(os.listdir(input_dir / cls))])

        for cls, filename in img_files:
            img_path = input_dir / cls / filename
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            edges = transform_fn(img_bgr)
            out_path = domain_dir / cls / filename
            cv2.imwrite(str(out_path), edges)
            processed += 1
            print(f'[{processed}/{total * len(transforms_map)}] {domain_name}/{cls}/{filename}', end='\r')

    print(f"\n预处理完成，输出目录: {output_dir}")
    for domain_name in transforms_map:
        count = sum(len(list((output_dir / domain_name / c).iterdir())) for c in classes)
        print(f"  {domain_name}: {count} 张")


if __name__ == '__main__':
    main()
