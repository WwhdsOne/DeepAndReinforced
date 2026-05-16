# ch5 — 误差反向传播法实现 TwoLayerNet

基于误差反向传播的手写数字识别（MNIST），用 NumPy 从零搭建两层神经网络。

## 项目结构

```
ch5/
├── affineLayer.py       # Affine 层（全连接 + 偏置），含 forward / backward
├── reluLayer.py         # ReLU 激活层，含 forward / backward
├── sigmoidLayer.py      # Sigmoid 激活层，含 forward / backward
├── softmaxLayer.py      # Softmax + SoftmaxWithLoss，含交叉熵损失
├── addLayer.py          # 加法层（供组合层使用）
├── multiLayer.py        # 乘法层（供组合层使用）
├── combineLayer.py      # 组合层示例（苹果橘子价格计算图）
├── train_nn.py          # TwoLayerNet 训练脚本
├── data/                # MNIST 数据缓存目录
└── ../common/
    ├── layers.py        # 层统一导入，提供 Affine / Relu / SoftmaxWithLoss 别名
    └── gradient.py      # numerical_gradient 数值微分
```

## 网络结构

```
输入层 (784) → Affine1 + ReLU → 隐藏层 (50) → Affine2 → 输出层 (10) → SoftmaxWithLoss
```

| 参数 | 形状 | 说明 |
|------|------|------|
| `W1` | `784 × 50` | 输入→隐藏层权重 |
| `b1` | `50` | 隐藏层偏置 |
| `W2` | `50 × 10` | 隐藏→输出层权重 |
| `b2` | `10` | 输出层偏置 |

## 训练流程

### 第一步：数据准备

6 万张 28×28 的灰度图，每张拉直为 784 维向量，逐行堆叠成 $60000 \times 784$ 的矩阵。像素值除以 255 归一化到 $[0, 1]$。

$$
X \in \mathbb{R}^{60000 \times 784}, \quad y \in \{0,\dots,9\}^{60000}
$$

### 第二步：Affine1 → ReLU

$$
Z_1 = X W_1 + b_1 \qquad (60000 \times 784) \cdot (784 \times 50) + (50) \;\to\; (60000 \times 50)
$$

$$
A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)
$$

### 第三步：Affine2

$$
Z_2 = A_1 W_2 + b_2 \qquad (60000 \times 50) \cdot (50 \times 10) + (10) \;\to\; (60000 \times 10)
$$

### 第四步：Softmax

对 $Z_2$ 的每一行做 softmax，得到每个样本属于 0~9 的概率分布：

$$
y_{nj} = \frac{\exp(Z_2[n, j])}{\sum_{k=0}^{9} \exp(Z_2[n, k])}
$$

$y_{nj}$ 表示第 $n$ 个样本属于类别 $j$ 的概率，$\sum_j y_{nj} = 1$。

### 第五步：交叉熵损失

令第 $n$ 个样本的真实标签为 $k_n \in \{0,\dots,9\}$。

**单个样本的交叉熵**（按标签索引计算）：

$$
E_n = -\ln(y_{n,k_n})
$$

$y_{n,k_n}$ 是第 $n$ 个样本在真实类别上的预测概率。概率越高（越接近 1），损失越小（越接近 0）。

**批量平均损失**（代码实际计算）：

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \ln(y_{n,k_n})
$$

即遍历 $N$ 个样本，取每个样本真实类别对应的预测概率，取 $\ln$ 后求平均再取负。

**等价形式（one-hot 写法，反向传播用）**：

令 $\mathbf{t}_n$ 为第 $n$ 个样本的 one-hot 向量（真实类别位置为 1，其余为 0），则：

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \sum_{j=0}^{9} t_{nj} \ln(y_{nj})
$$

内层 $\sum_j$ 中只有 $j = k_n$ 时 $t_{nj} = 1$，其余为 0，因此等价于 $-\frac{1}{N}\sum_n \ln(y_{n,k_n})$。

**数值稳定性**：$y_{nj}$ 加 $\varepsilon = 10^{-7}$ 防止 $\ln(0)$。

**返回值**：标量，代表该批次的平均损失。

### 第六步：反向传播

从损失出发，对 softmax + 交叉熵联合求导（可化简为预测概率减真实标签）：

| 层 | 反向传播公式 |
|----|-------------|
| SoftmaxWithLoss | $$dZ_2 = \frac{y - t}{N}$$ |
| Affine2 | $$dA_1 = dZ_2 \cdot W_2^T \qquad dW_2 = A_1^T \cdot dZ_2 \qquad db_2 = \sum_{rows} dZ_2$$ |
| ReLU | $$dZ_1 = dA_1 \odot \mathbf{1}[Z_1 > 0]$$  — 前向时 $\le 0$ 的位置梯度置 0 |
| Affine1 | $$dX = dZ_1 \cdot W_1^T \qquad dW_1 = X^T \cdot dZ_1 \qquad db_1 = \sum_{rows} dZ_1$$ |

其中 $\sum_{rows}$ 表示沿行方向求和（对 batch 维度求和）。

**梯度形状一览**：

| 梯度 | 形状 | 来源 |
|------|------|------|
| $dZ_2$ | $N \times 10$ | $y - t$ |
| $dW_2$ | $50 \times 10$ | $A_1^T \cdot dZ_2$ |
| $db_2$ | $10$ | $\sum_{rows} dZ_2$ |
| $dZ_1$ | $N \times 50$ | $dA_1 \odot \mathbf{1}[Z_1 > 0]$ |
| $dW_1$ | $784 \times 50$ | $X^T \cdot dZ_1$ |
| $db_1$ | $50$ | $\sum_{rows} dZ_1$ |

### 第七步：更新参数

$$
W_1 \leftarrow W_1 - \eta \cdot dW_1, \quad b_1 \leftarrow b_1 - \eta \cdot db_1
$$
$$
W_2 \leftarrow W_2 - \eta \cdot dW_2, \quad b_2 \leftarrow b_2 - \eta \cdot db_2
$$

其中 $\eta$ 为学习率。然后回到第二步，重复迭代。

## 运行

```bash
cd experiments/fish-book/ch5
python3 train_nn.py
```

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_size` | 784 | 输入维度（28×28） |
| `hidden_size` | 50 | 隐藏层神经元数 |
| `output_size` | 10 | 输出类别数（0~9） |
| `iters_num` | 10000 | 总迭代次数 |
| `batch_size` | 100 | 每批样本数 |
| `learning_rate` | 0.1 | 学习率 |
| `weight_init_std` | 0.01 | 权重初始化标准差（高斯分布） |
