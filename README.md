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
.
├── AGENTS.md
├── pyproject.toml
├── experiments/
│   └── pic-classify/
│       ├── README.md
│       ├── src/pic_classify/
│       └── tests/
└── pic-classify/
```

说明：

- `experiments/` 放各个独立算法实验
- 每个实验目录内维护自己的代码、测试和说明
- `pic-classify/` 目前保留旧数据目录，但当前代码不再依赖它

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
