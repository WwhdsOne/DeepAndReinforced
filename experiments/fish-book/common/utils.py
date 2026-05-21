"""卷积相关的通用张量工具。"""

import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """将输入批量展开为适合卷积计算的二维矩阵。

    使用 as_strided 滑动窗口实现，避免 Python 循环，性能远优于逐元素遍历。
    """
    n, c, h, w = input_data.shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode="constant")

    # 使用 stride tricks 创建滑动窗口视图
    # 形状: (N, C, out_h, out_w, filter_h, filter_w)
    # 每个 (out_h, out_w) 位置对应 filter_h×filter_w 的滑动窗口
    shape = (n, c, out_h, out_w, filter_h, filter_w)
    strides = (
        img.strides[0],              # N 维度步幅不变
        img.strides[1],              # C 维度步幅不变
        stride * img.strides[2],      # 高度方向每次跳 stride 行
        stride * img.strides[3],      # 宽度方向每次跳 stride 列
        img.strides[2],              # 滤波器内：连续行
        img.strides[3],              # 滤波器内：连续列
    )

    col_view = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    # 转置为 (N, out_h, out_w, C, filter_h, filter_w) → 重塑为 (N*out_h*out_w, C*filter_h*filter_w)
    col = col_view.transpose(0, 2, 3, 1, 4, 5).reshape(n * out_h * out_w, -1)
    return np.ascontiguousarray(col)

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 重塑为 (N, C, filter_h, filter_w, out_h, out_w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)

    # 使用 as_strided 将 img 映射为与 col 同形的视图，一次性完成 scatter-add
    shape = (N, C, filter_h, filter_w, out_h, out_w)
    strides = (
        img.strides[0],              # N
        img.strides[1],              # C
        img.strides[2],              # filter_h: 连续行
        img.strides[3],              # filter_w: 连续列
        stride * img.strides[2],      # out_h: 跳 stride 行
        stride * img.strides[3],      # out_w: 跳 stride 列
    )
    img_view = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    img_view += col

    if pad > 0:
        return img[:, :, pad:H + pad, pad:W + pad]
    return img

def load_mnist(data_dir: str) -> tuple:
    """加载 MNIST 数据集（使用 npz 格式）。"""
    import urllib.request
    import ssl
    from pathlib import Path

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "mnist.npz"

    if not cache.exists():
        urls = [
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            "https://github.com/fgnt/mnist/raw/master/mnist.npz",
        ]
        ctx = ssl._create_unverified_context()
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                    cache.write_bytes(resp.read())
                break
            except Exception:
                continue
        else:
            raise RuntimeError("MNIST 下载失败，请手动放置 mnist.npz")

    with np.load(cache) as data:
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]
