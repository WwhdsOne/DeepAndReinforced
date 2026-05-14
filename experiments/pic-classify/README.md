# pic-classify

基于 `PyTorch + torchvision` 的 CIFAR-10 图片分类实验。

## 目录结构

```text
pic-classify/
├── README.md
├── src/pic_classify/
└── tests/
```

## 运行方式

在仓库根目录执行：

```bash
uv sync --group dev
uv run pic-classify-train
uv run pic-classify-predict path/to/image.png
```

## 默认路径

- 数据集默认下载到 `experiments/pic-classify/data/`
- 模型默认保存到 `experiments/pic-classify/artifacts/cifar10_cnn.pt`

## 可调整参数

### 1. 训练入口 `pic-classify-train`

命令：

```bash
uv run pic-classify-train [options]
```

支持的参数如下：

- `--data-dir PATH`
  - CIFAR-10 数据集下载和读取目录。
  - 默认值：`experiments/pic-classify/data/`
- `--output PATH`
  - 训练完成后模型保存路径。
  - 默认值：`experiments/pic-classify/artifacts/cifar10_cnn.pt`
- `--epochs INT`
  - 训练轮数。
  - 默认值：`5`
- `--batch-size INT`
  - 每个 batch 的样本数。
  - 默认值：`128`
- `--learning-rate FLOAT`
  - Adam 优化器学习率。
  - 默认值：`0.001`
- `--optimizer {adam,sgd}`
  - 训练时使用的优化器。
  - 默认值：`adam`
- `--num-workers INT`
  - `DataLoader` 的加载进程数。
  - 默认值：`0`

训练时会显示 batch 级进度条，并在每个 epoch 结束后输出一次 `train_loss` 和 `test_accuracy`。

示例：

```bash
uv run pic-classify-train --epochs 20 --batch-size 64 --learning-rate 0.0005 --optimizer sgd
```

### 2. 预测入口 `pic-classify-predict`

命令：

```bash
uv run pic-classify-predict path/to/image.png [options]
```

支持的参数如下：

- `image`
  - 待预测图片路径，必填。
- `--model PATH`
  - 加载的模型文件路径。
  - 默认值：`experiments/pic-classify/artifacts/cifar10_cnn.pt`
- `--top-k INT`
  - 输出概率最高的前 `k` 个类别。
  - 默认值：`3`

示例：

```bash
uv run pic-classify-predict ./demo.png --top-k 5
```

## 训练增强与归一化说明

### 训练增强（Data Augmentation）

训练增强通过对训练图像做随机变换，人为增加数据多样性，从而抑制过拟合、提升模型泛化能力。当前 `build_train_transform()` 包含两种增强：

| 变换 | 说明 |
|---|---|
| `RandomHorizontalFlip()` | 以 50% 概率对图像做水平翻转。CIFAR-10 中许多物体（如飞机、马）水平翻转后仍合法，此操作可有效增加样本多样性。 |
| `RandomCrop(32, padding=4)` | 先对图像四周各填充 4 像素（共 36×36），再随机裁剪回 32×32。使模型对物体的平移和局部遮挡更鲁棒。 |

> 验证/预测阶段不使用增强，仅做 `Resize` + `ToTensor()` + 归一化，保证结果确定性。

### 归一化参数（Normalization）

`transforms.Normalize(mean, std)` 将每个通道的像素值按以下公式归一化到近似零均值、单位方差：
$$
\text{output} = \frac{\text{input} - \text{mean}}{\text{std}}
$$

其中标准差 $\text{std}$（按通道独立计算）为：
$$
\text{std} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \text{mean})^2}
$$

- $x_i$：单个像素值（某个通道）
- $\text{mean}$：该通道的均值
- $N$：该通道的像素总数
- $\sum$：对所有像素求和

CIFAR-10 的均值和标准差是从整个训练集统计得到的，写入 `data.py` 中的常量：

```
mean = (0.4914, 0.4822, 0.4465)   # (R, G, B)
std  = (0.2470, 0.2435, 0.2616)   # (R, G, B)
```

使用这些预计算统计量（而非每次动态计算）的原因是：
- 训练集和验证集必须使用相同的均值/标准差，否则分布不一致会影响模型性能；
- 与 torchvision 官方 CIFAR-10 示例保持一致，便于复现结果。

---

## 当前固定配置

下面这些参数目前写死在代码中，不通过命令行暴露：

- 数据集：`torchvision.datasets.CIFAR10(download=True)`
- 训练增强：
  - `RandomHorizontalFlip()`
  - `RandomCrop(32, padding=4)`
- 验证/预测预处理：
  - `Resize((32, 32))`
  - `ToTensor()`
  - 按 CIFAR-10 均值方差归一化
- 归一化参数：
  - `mean = (0.4914, 0.4822, 0.4465)`
  - `std = (0.2470, 0.2435, 0.2616)`
- 模型结构：
  - 3 个卷积块
  - 卷积通道数依次为 `32 -> 64 -> 128`
  - `AdaptiveAvgPool2d((1, 1))`
  - 最终全连接层输出 `10` 类
- 损失函数：`CrossEntropyLoss`
- 优化器：默认 `Adam`，也可通过 `--optimizer sgd` 切换为 `SGD`
- 运行设备：自动选择 `cuda`，否则使用 `cpu`

## 说明

- 训练时会自动下载 CIFAR-10 数据集。
- 预测时输入图片会先转换为 RGB，再按与验证阶段一致的方式预处理。
- 模型文件里会同时保存 `class_names`，用于预测时输出类别名称。
