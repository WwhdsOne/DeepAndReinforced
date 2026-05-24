# ch8 使用指南

## 文件结构

```
ch8/
├── README.md                     # 模型架构说明（VGG / GoogLeNet / ResNet）
├── GUIDE.md                      # 本文件：使用指南
├── vgg16_architecture.svg        # VGG16 架构图
├── train.py                      # 训练入口
├── tiny_imagenet.py              # Tiny ImageNet 加载器
└── models/
    ├── __init__.py               # 模型注册中心
    └── vgg.py                    # VGG 定义
```

数据集已就位在 `experiments/fish-book/data/tiny-imagenet-200/`，包含：

| 集 | 数量 | 说明 |
|:---|:----|:-----|
| 训练集 | 100,000 张（200 类 × 500 张） | 64×64 RGB |
| 验证集 | 10,000 张 | 带标注 |
| 测试集 | 10,000 张 | 无标注 |

## 训练命令

```bash
# 进入 ch8 目录
cd experiments/fish-book/ch8

# ── VGG16 默认训练 ──
uv run python train.py

# ── VGG19 ──
uv run python train.py --model VGG19

# ── 快速验证（小轮数） ──
uv run python train.py --model VGG11 --epochs 5

# ── 自定义参数 ──
uv run python train.py \
    --model VGG16 \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.01
```

### 参数说明

| 参数 | 默认值 | 说明 |
|:----|:------|:-----|
| `--model` | VGG16 | 可选 VGG11 / VGG13 / VGG16 / VGG19 |
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 128 | 批次大小 |
| `--lr` | 0.01 | 学习率 |
| `--momentum` | 0.9 | SGD 动量 |
| `--weight-decay` | 5e-4 | 权重衰减（L2 正则化） |
| `--device` | auto | cuda / cpu / auto |
| `--data-root` | ../data | 数据集所在目录 |
| `--drop-out` | 0.5 | 随机放弃一批神经元 |

设备选择：
- M 芯片 Mac 会自动使用 MPS（Metal）加速
- 无 GPU 则自动回退 CPU

## 训练输出

```
训练记录保存到: ch8/artifacts/<模型名>_tiny_imagenet_history.json
模型权重保存到: ch8/artifacts/<模型名>_tiny_imagenet.pth
```

训练过程中每轮打印一次摘要：

```
Epoch  Train Loss  Train Acc  Val Loss  Val Acc  Time
------------------------------------------------------------
    1      2.3456     45.23%    1.9876    52.10%    25s  *
    2      1.8765     55.12%    1.6543    58.30%    25s
    3      1.5432     62.45%    1.4321    63.80%    25s  *
```

带 `*` 的行表示验证准确率刷新了最佳记录。

## 添加新模型（以 ResNet 为例）

```bash
ch8/
└── models/
    ├── __init__.py       ← 在这里注册
    ├── vgg.py            ← VGG
    └── resnet.py         ← 新建
```

**第一步**：创建 `models/resnet.py`，定义 `ResNet18` 类（继承 `nn.Module`）。

**第二步**：在 `models/__init__.py` 注册：

```python
from .resnet import ResNet18
MODEL_REGISTRY["ResNet18"] = lambda **kw: ResNet18(**kw)
```

**第三步**：直接训练：

```bash
uv run python train.py --model ResNet18
```

`train.py` 不需要任何修改。

## 数据加载器用法（单独调用）

```python
from tiny_imagenet import TinyImageNet, get_loaders

# 方式一：直接获取 DataLoader
train_loader, val_loader = get_loaders("data", batch_size=128)

# 方式二：自定义 transform
from torchvision import transforms
custom_transform = transforms.Compose([...])
dataset = TinyImageNet("data", split="train", transform=custom_transform)
```
