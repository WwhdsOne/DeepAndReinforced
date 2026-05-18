# DeepAndReinforced

研究生入学前的算法学习与相关实验仓库。

当前仓库以根目录统一的 `uv` 工程管理 Python 依赖、脚本入口和测试。每个算法实验单独放在 `experiments/` 下，保持结构独立，但共用同一套依赖管理。

## 环境准备

要求：

- Python 3.11 及以上
- 已安装 `uv`

安装依赖：

```bash
uv sync --group dev
```

## 仓库结构

```text
.                                    # 项目根目录
├── AGENTS.md                        # AI代理配置文件
├── pyproject.toml                   # Python项目配置和依赖管理
├── experiments/                     # 实验代码目录
│   ├── pic-classify/                # 图片分类实验
│   └── fish-book/                   # 鱼书学习实验
│       ├── ch3/                     # 第3章：神经网络基础
│       ├── ch4/                     # 第4章：梯度计算与优化
│       ├── ch5/                     # 第5章：神经网络层实现
│       └── common/                  # 通用工具函数
└── learning-logs/                   # 学习日志目录
```

说明：

- `experiments/` 放各个独立算法实验，每个实验目录内维护自己的代码、测试和说明
- `fish-book/` 是《深度学习入门：基于Python的理论与实现》的学习实验，按章节组织
- `learning-logs/` 记录每日学习进度和总结，格式为 `YYYY-MM-DD.md`

## pic-classify

`pic-classify` 是一个基于 `PyTorch + torchvision` 的 CIFAR-10 图片分类实验。

特性：

- 使用 `torchvision.datasets.CIFAR10(download=True)` 自动下载数据
- 提供训练入口和单图预测入口
- 数据默认保存到 `experiments/pic-classify/data/`
- 模型权重默认保存到 `experiments/pic-classify/artifacts/`
- 实验代码位于 `experiments/pic-classify/src/pic_classify/`
- 实验测试位于 `experiments/pic-classify/tests/`

### 训练

```bash
uv run pic-classify-train --epochs 5
```

可选参数示例：

```bash
uv run pic-classify-train \
  --data-dir experiments/pic-classify/data \
  --output experiments/pic-classify/artifacts/cifar10_cnn.pt \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --epochs 5
```

### 预测

```bash
uv run pic-classify-predict path/to/image.png
```

指定模型与返回 Top-K：

```bash
uv run pic-classify-predict path/to/image.png \
  --model experiments/pic-classify/artifacts/cifar10_cnn.pt \
  --top-k 3
```

## fish-book

`fish-book` 是《深度学习入门：基于Python的理论与实现》（俗称“鱼书”）的学习实验代码，按章节组织。

### 内容概览

- **第3章**：神经网络基础实现，包括感知机、三层神经网络、激活函数可视化
- **第4章**：梯度计算与优化，包括数值梯度、梯度下降、简单神经网络训练
- **第5章**：神经网络层实现，包括乘法层、加法层、ReLU层、Sigmoid层、Softmax层、Affine层等
- **通用工具**：`common/` 目录包含梯度计算、层定义等通用函数

### 运行示例

```bash
# 运行第5章乘法层示例
uv run python experiments/fish-book/ch5/multiLayer.py

# 运行第5章神经网络训练
uv run python experiments/fish-book/ch5/train_nn.py
```

### 学习笔记

每日学习进度和总结记录在 `learning-logs/` 目录下，格式为 `YYYY-MM-DD.md`。

## 测试

运行测试：

```bash
uv run --group dev pytest -q
```

## 后续扩展建议

- 新增实验时，优先建立 `experiments/<实验名>/`
- 每个实验目录内部再维护自己的 `src/`、`tests/`、`README.md`
- 新增命令行入口时，在根目录 `pyproject.toml` 中统一注册
- 尽量复用同一个 `uv` 环境，不要为子目录单独维护依赖
- 学习笔记统一记录在 `learning-logs/` 目录下，按日期组织
- 鱼书实验按章节组织在 `experiments/fish-book/` 下，便于学习追踪
