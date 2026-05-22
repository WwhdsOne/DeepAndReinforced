# ch8 — 深度学习

本目录涵盖第 8 章涉及的经典卷积神经网络架构，包括 VGG、GoogLeNet、ResNet 等。每个架构独立成节，包含设计思路、核心原理与关键代码说明。

---

## VGG

**VGG**（Visual Geometry Group）是牛津大学视觉几何组在 2014 年提出的经典卷积神经网络架构，发表于论文 *"Very Deep Convolutional Networks for Large-Scale Image Recognition"*（Simonyan & Zisserman, 2014）。

VGG 在 2014 年 ImageNet 竞赛中定位任务取得第一名、分类任务第二名，其核心贡献在于**证明了网络深度对性能的提升作用**，并确立了**小卷积核堆叠**的设计范式。

### 架构图解

以下是 VGG16（配置 D）的网络结构图：

![vgg16_architecture](https://wwhds-markdown-image.oss-cn-beijing.aliyuncs.com/vgg16_architecture_fix.svg)

### 核心设计原则

#### 1. 小卷积核堆叠

VGG 统一使用 **3×3 卷积核**，摒弃了 AlexNet 中的大卷积核（11×11, 5×5）。**两个 3×3 堆叠**（stride=1, pad=1）等效于**一个 5×5**的感受野，**三个 3×3 堆叠**等效于**一个 7×7**，但：

| 特性 | 单个 7×7 | 三个 3×3 堆叠 |
|:---|:--------:|:-----------:|
| 参数量 | $7^2 C^2 = 49 C^2$ | $3 \times 3^2 C^2 = 27 C^2$ |
| 非线性层数 | 1 层 ReLU | 3 层 ReLU |
| 正则化效果 | 无 | 隐含正则化（更深的非线性变换） |

> 参数量减少约 **45%**，且更多的非线性层提升了判别能力。

#### 2. 深度的系统探索

论文系统性地比较了从 11 层到 19 层的不同深度配置，证明了**增加深度可以提升性能**，这一发现在当时极具影响力。

#### 3. 结构简洁统一

- 所有卷积层：`3×3` 核，`pad=1`，`stride=1`
- 所有池化层：`2×2` 窗口，`stride=2`
- 所有全连接层后接 ReLU + Dropout（$p = 0.5$）

每个池化层后特征图尺寸减半、通道数加倍（从 64 递增到 512）。

#### 4. 优化器

- 带动量的 SGD（SGD+Momentum），泛化性能在当前场景下会相对于Adam会好一些
- 默认 **momentum=0.9**
- 默认 **weight_decay=5e-4**,weight_decay就是 L2 正则化

### 运行结果

运行指令：

> !python train.py --model VGG11 --epochs 15

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
| :---- | :--------- | :-------- | :------- | :------ | :--- |
| 5     | 3.7925     | 15.54%    | 3.7071   | 16.91%  | 81s  |
| 10    | 2.9104     | 31.19%    | 2.9020   | 31.89%  | 80s  |
| 15    | 2.3334     | 42.92%    | 2.5888   | 38.77%  | 81s  |

使用的是**colab免费T4**

---

## GoogLeNet

> 待补充

---

## ResNet

> 待补充
