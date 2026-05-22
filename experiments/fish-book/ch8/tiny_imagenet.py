"""
Tiny ImageNet-200 数据加载器。

用法：
    from tiny_imagenet import TinyImageNet, get_loaders

    train_loader, val_loader = get_loaders("../data", batch_size=128)
"""
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TinyImageNet(Dataset):
    """Tiny ImageNet-200 数据集，支持 train / val 两种 split。"""

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root) / "tiny-imagenet-200"
        self.transform = transform
        self._build_label_map()

        if split == "train":
            self._load_train()
        elif split == "val":
            self._load_val()
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    # ── 内部方法 ──────────────────────────────────

    def _build_label_map(self):
        """从 wnids.txt 构建 wnid → 0~199 的映射。"""
        wnids_path = self.root / "wnids.txt"
        wnids = wnids_path.read_text().strip().splitlines()
        self._wnid2label = {w: i for i, w in enumerate(wnids)}

    def _load_train(self):
        self.images, self.labels = [], []
        for cls_dir in sorted(self.root.glob("train/*")):
            wnid = cls_dir.name
            label = self._wnid2label[wnid]
            img_dir = cls_dir / "images"
            if img_dir.is_dir():
                for img_path in sorted(img_dir.glob("*.JPEG")):
                    self.images.append(img_path)
                    self.labels.append(label)

    def _load_val(self):
        self.images, self.labels = [], []
        annot_path = self.root / "val/val_annotations.txt"
        for line in annot_path.read_text().strip().splitlines():
            parts = line.split()
            img_path = self.root / "val/images" / parts[0]
            wnid = parts[1]
            self.images.append(img_path)
            self.labels.append(self._wnid2label[wnid])

    # ── 公开方法 ──────────────────────────────────

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def class_names(self):
        """返回 200 个类别的英文名称列表。"""
        if hasattr(self, "_class_names"):
            return self._class_names
        words_path = self.root / "words.txt"
        wnid2word = {}
        for line in words_path.read_text().strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                wnid2word[parts[0]] = parts[1].split(",")[0].strip()
        self._class_names = [
            wnid2word.get(w, w)
            for w in self.root.joinpath("wnids.txt").read_text().strip().splitlines()
        ]
        return self._class_names


def ensure_dataset(root):
    """确保 tiny-imagenet-200 数据集存在，不存在则自动下载。"""
    data_dir = Path(root) / "tiny-imagenet-200"
    if (data_dir / "wnids.txt").exists():
        print(f"数据集已存在: {data_dir}")
        return

    Path(root).mkdir(parents=True, exist_ok=True)
    print("正在下载 Tiny ImageNet-200...")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = Path(root) / "tiny-imagenet-200.zip"
    urlretrieve(url, zip_path)

    print("下载完成，正在解压...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)
    zip_path.unlink()  # 删掉压缩包，省空间
    print("数据集已就绪")


def get_transform(train=True):
    """训练/验证的标准预处理管道。"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def get_loaders(data_root, batch_size=128, num_workers=4):
    """返回 (train_loader, val_loader)。"""
    ensure_dataset(data_root)
    train_set = TinyImageNet(data_root, split="train",
                             transform=get_transform(train=True))
    val_set = TinyImageNet(data_root, split="val",
                           transform=get_transform(train=False))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
