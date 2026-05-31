# Lecture 13 Exercise 1 — 情感分析（Sentiment Analysis）

## 任务描述

基于 LSTM 的情感五分类模型，对电影评论进行情感分析。

**类别**：
| 标签 | 含义 |
|------|------|
| 0 | negative |
| 1 | somewhat negative |
| 2 | neutral |
| 3 | somewhat positive |
| 4 | positive |

## 模型架构

![lecture-13-arch](https://wwhds-markdown-image.oss-cn-beijing.aliyuncs.com/lecture-13-arch-fix.svg)

- **Embedding**: `vocab_size → 64`，`padding_idx=0`
- **Bi-LSTM**: hidden=128, num_layers=2, dropout=0.5, batch_first
- **FC**: `hidden_size * 2 → num_classes(5)`
- **损失函数**: `CrossEntropyLoss`
- **优化器**: `Adam(lr=0.001)`

## 数据集

- 来源：Kaggle — [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
- 总样本：156,060
  - 训练集：124,848（80%）
  - 验证集：31,212（20%）
- Batch size: 512

## 训练环境

- **设备**: NVIDIA T4 (Colab)
- **框架**: PyTorch
- **Epochs**: 20

## 训练结果

| Epoch | Loss | Accuracy |
|-------|------|----------|
|  1 | 1.3006 | 51.17% |
|  2 | 1.2218 | 52.12% |
|  3 | 1.1655 | 53.97% |
|  4 | 1.1050 | 55.56% |
|  5 | 1.0404 | 57.95% |
|  6 | 0.9794 | 60.59% |
|  7 | 0.9180 | 61.93% |
|  8 | 0.8713 | 62.61% |
|  9 | 0.8333 | 63.41% |
| 10 | 0.7938 | 64.32% |
| 11 | 0.7679 | 64.32% |
| 12 | 0.7396 | 64.73% |
| 13 | 0.7157 | 65.39% |
| 14 | 0.6966 | 65.42% |
| 15 | 0.6808 | 64.41% |
| 16 | 0.6709 | 65.51% |
| 17 | 0.6443 | 65.51% |
| 18 | 0.6300 | 65.50% |
| 19 | 0.6188 | 65.39% |

## 分析

- **收敛趋势**：Loss 从 1.30 持续下降到 0.62，模型稳定收敛
- **最佳准确率**：约 **65.5%**（Epoch 16-18）
- **五分类随机基线**：20%，模型远超随机水平
- **瓶颈**：到 10 epoch 后准确率增长趋缓，Loss 仍在下降但 Acc 提升不明显，可能出现轻微过拟合或模型容量不足

## 代码

- `experiments/liuerdaren/lecture-13-exercise-1.py`
