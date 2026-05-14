# neural_network

本实验使用 NumPy 实现多层感知机，并提供一个基于 MNIST 的手写数字识别小项目。

## 项目结构

- `src/neural_network/multilayer_perceptron.py`：核心模型实现
- `src/neural_network/mnist_dataset.py`：MNIST 数据加载与图片预处理
- `src/neural_network/mnist.py`：训练、预测、评估入口
- `src/neural_network/visualize.py`：训练过程可视化（生成 HTML 仪表盘）
- `data/`：MNIST 下载目录
- `artifacts/`：模型文件、训练历史、可视化页面输出目录

## 安装依赖

```bash
uv sync --group dev
```

## 快速开始

训练模型：

```bash
uv run mnist-train --epochs 20 --learning-rate 0.1
```

预测单张图片：

```bash
uv run mnist-predict path/to/image.png
```

评估已保存模型：

```bash
uv run mnist-evaluate
```

生成训练过程可视化页面：

```bash
uv run mnist-visualize
# 然后用浏览器打开 artifacts/mnist_mlp.html
```

## 命令说明

### `mnist-train`

训练 MNIST 模型并保存权重。

训练流程：

1. 从 `--data-dir` 加载 MNIST 数据集（缓存不存在时自动下载）
2. 按 `--limit-train` 和 `--limit-test` 截断训练/测试集
3. 构建三层感知机：输入层 `784` → 隐藏层 `--hidden-size` → 输出层 `10`
4. 初始化权重（Xavier 均匀分布）
5. 使用 `mini-batch + Adam` 训练 `--epochs` 轮，每轮计算交叉熵损失
6. 在测试集上评估准确率
7. 将模型权重和结构保存到 `--output`

常用参数：

- `--data-dir`：MNIST 数据下载目录，默认 `data/`（自动解析到项目 data 目录）
- `--output`：模型保存路径，默认 `artifacts/mnist_mlp.npz`（自动解析到项目 artifacts 目录）
- `--epochs`：训练轮数，默认 `20`
- `--learning-rate`：学习率，默认 `0.001`
- `--hidden-size`：隐藏层神经元数量，默认 `256`
- `--batch-size`：每个 mini-batch 的样本数，默认 `64`
- `--optimizer`：优化器，默认 `adam`，也支持 `sgd`
- `--limit-train`：训练集截断数量，默认 `60000`
- `--limit-test`：测试集截断数量，默认 `2000`

### `mnist-visualize`

读取训练历史 JSON，生成包含损失曲线和准确率曲线的独立 HTML 仪表盘。

参数：

- `history`（可选）：训练历史 JSON 文件路径，默认 `artifacts/mnist_mlp.history.json`
- `-o / --output`：输出 HTML 文件路径，默认与 history 同名 `.html`

### `mnist-predict`

对一张手写数字图片进行识别。

参数：

- `image`（必填）：待识别的图片路径
- `--model`：模型文件路径，默认 `artifacts/mnist_mlp.npz`（自动解析到项目 artifacts 目录）

处理流程（代码自动完成，无需手动操作）：

1. 读取图片并转换为灰度图
2. 缩放到 `28x28` 像素
3. 像素值归一化到 `0~1` 范围 `max - min`缩放
4. 反色处理（MNIST 是黑底白字，如果输入是白底黑字会自动反转）
5. 展平为 `784` 维特征向量
6. 加载模型，前向传播，输出预测数字

### `mnist-evaluate`

加载已保存模型并在测试集上评估准确率。

参数：

- `--data-dir`：MNIST 数据目录，默认 `data/`（自动解析到项目 data 目录）
- `--model`：模型文件路径，默认 `artifacts/mnist_mlp.npz`（自动解析到项目 artifacts 目录）
- `--limit-test`：测试集截断数量，默认 `2000`

## 模型说明

核心模型是 `MultilayerPerceptron`，默认网络结构为三层：

```
输入层 (784) → 隐藏层 (默认 256) → 输出层 (10)
```

- 隐藏层激活函数：`ReLU`
- 输出层激活函数：`softmax`
- 损失函数：交叉熵
- 优化算法：`mini-batch + Adam`
- 权重初始化：Xavier 均匀分布（`epsilon = sqrt(6 / (in + out))`）
- 这是实验性质实现，暂未加入正则化和早停

### 偏置值为什么这么设计

本项目没有单独保存 `b`，而是把偏置并入权重矩阵第一列。做法是：

1. 在输入特征前拼一列全 1
2. 让权重矩阵多出一列，专门乘这列 1
3. 前向传播时直接用一次矩阵乘法完成线性变换

这样设计的原因是：

- 统一数学形式，计算时只需要处理矩阵乘法
- 不必单独维护 `b` 和 `db`
- 反向传播时更容易向量化实现

以一层网络为例，若输入是 `x = [x1, x2, ..., xn]`，加偏置后变成：

```text
[1, x1, x2, ..., xn]
```

对应的权重写成：

```text
[b, w1, w2, ..., wn]
```

那么该神经元的线性部分就是：

```text
z = b + w1*x1 + w2*x2 + ... + wn*xn
```

在代码里，矩阵形状会是下面这样：

- 原始输入：`(m, 784)`
- 加上偏置列后：`(m, 785)`
- 第一层权重：`(256, 785)`
- 隐藏层输出：`(m, 256)`
- 再加偏置列后：`(m, 257)`
- 第二层权重：`(10, 257)`
- 最终输出概率：`(m, 10)`

反向传播里使用 `weights[:, 1:]`，是因为第一列对应偏置，不属于上一层激活值，计算上一层误差时要把它排除掉。

## 默认超参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `50` | 训练轮数 |
| `--learning-rate` | `0.001` | 学习率 |
| `--hidden-size` | `256` | 隐藏层神经元个数 |
| `--batch-size` | `64` | mini-batch 大小 |
| `--optimizer` | `adam` | 优化器 |
| `--limit-train` | `60000` | 训练集样本数（MNIST 共 60000） |
| `--limit-test` | `2000` | 测试集样本数（MNIST 共 10000） |

## 低准确率总结

如果训练结果不理想，通常是因为下面这些原因叠加：

- 隐藏层使用 `sigmoid`，在 MNIST 这类任务上更容易梯度饱和
- 仅一层隐藏层，模型容量偏小
- 全批量梯度下降收敛慢，对学习率更敏感
- 训练集样本数过少时，模型更容易欠拟合

本次记录到的较低准确率结果如下：

- 训练完成，最后一次损失：`1.795541`
- 测试集准确率：`0.5855`
- 运行时命令：`uv run mnist-train --epochs 50 --learning-rate 0.1 --limit-train 60000`

对应的改进方向是：

- 把隐藏层激活改为 `ReLU`
- 使用 `mini-batch + Adam`
- 增加隐藏层宽度，必要时再加一层隐藏层
- 训练时尽量使用完整的 MNIST 训练集

优化后结果为：

- 训练完成，最后一次损失：`0.051514 `
- 测试集准确率：`0.9615`

## 输出文件

- 数据会下载到 `data/` 目录
- 模型、训练历史、可视化页面会保存到 `artifacts/` 目录

## 数据准备

脚本会优先读取本地缓存文件 `data/mnist.npz`。

如果缓存不存在，会自动尝试从以下镜像依次下载（任一成功即停止）：

1. `storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`
2. `github.com/fgnt/mnist/raw/master/mnist.npz`（备用镜像）

如果所有镜像都不可用，可以手动把 `mnist.npz` 放到 `data/` 目录下。

## 注意事项

- 训练大数据时会比较慢
- 单张图片识别效果取决于图片是否接近 MNIST 风格
- 如果图片背景和手写笔迹颜色与 MNIST 相反，脚本会自动做一次反色处理
