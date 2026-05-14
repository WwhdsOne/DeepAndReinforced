"""MNIST 数据集加载与预处理工具。"""

from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MNIST_NPZ_URLS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "https://github.com/fgnt/mnist/raw/master/mnist.npz",
]


def load_mnist_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 MNIST 数据集，优先使用本地缓存。"""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return _load_with_npz(root)


def load_image_as_mnist_features(image_path: str | Path) -> np.ndarray:
    """读取单张图片并转换成 MNIST 特征向量。"""
    image = Image.open(image_path).convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.float32)

    if array.max() > 1.0:
        array /= 255.0

    array = 1.0 - array
    return array.reshape(1, -1)


def _load_with_npz(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从本地 mnist.npz 读取数据，不存在时自动下载。"""
    cache_path = root / "mnist.npz"
    if not cache_path.exists():
        _download_file_insecure(MNIST_NPZ_URLS, cache_path)

    with np.load(cache_path) as payload:
        x_train = _normalize_images(payload["x_train"])
        y_train = payload["y_train"].astype(np.int64)
        x_test = _normalize_images(payload["x_test"])
        y_test = payload["y_test"].astype(np.int64)

    return x_train, y_train, x_test, y_test


def _normalize_images(images: np.ndarray) -> np.ndarray:
    """把原始图像统一成 0~1 范围，并反色为手写笔迹更亮。"""
    images = images.astype(np.float32) / 255.0
    images = 1.0 - images
    return images.reshape(images.shape[0], -1)


def _download_file_insecure(urls: list[str], destination: Path) -> None:
    """尝试从多个镜像下载，任一成功即停止。"""
    context = ssl._create_unverified_context()
    errors = []
    for url in urls:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, context=context, timeout=120) as response:
                destination.write_bytes(response.read())
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if destination.exists():
                destination.unlink()
    raise RuntimeError(
        "MNIST 下载失败，所有镜像均不可用：\n" + "\n".join(errors)
    )
