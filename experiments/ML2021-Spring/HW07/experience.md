# HW07 BERT 问答作业 — 经验总结

## 一、参数调优踩坑

### 1. 学习率不是越大越好

大模型（324M）对 lr 非常敏感：

| 学习率 | 表现 |
|---|---|
| 1e-4 | Epoch 1 训练 acc 从 0.73 一路降到 0.68，Epoch 2 直接崩溃（loss=5.27, acc=0） |
| 5e-5 | 训练稳定，3 epoch 验证 acc 达到 0.783 |

**结论**：BERT-base 用 1e-4 没问题，324M 大模型降到 5e-5 更安全。

### 2. Warmup 占比不宜过高

| warmup 步数 | 占总步数比例 | 效果 |
|---|---|---|
| 500 | 20% | lr 到达峰值时已跑完 1/5 训练，峰值过高导致过冲 |
| 200 | 8% | 快速进入有效学习率区间，训练更稳定 |

常见范围是总步数的 **5~10%**。

### 3. T_max 必须与实际步数对齐

余弦退火的 `T_max` 如果设太大（如固定 10000），lr 几乎不衰减，等于没用。

正确做法：`T_max = total_optimizer_steps - warmup_steps`

梯度累积时注意：`total_optimizer_steps = len(train_set) / (batch_size × gradient_accumulation_steps) × num_epoch`

### 4. batch_size 与迭代次数的权衡

固定 epoch 下，加大 batch_size 会减少梯度更新次数。epoch=1 时尤其危险——更新太少模型学不充分。

梯度累积是折中方案：物理 batch=4 省显存，累积 8 步后更新，等效 batch=32。

## 二、数据预处理

### 1. doc_stride 必须小于 max_paragraph_len

`doc_stride = max_paragraph_len` 意味着窗口无重叠，答案跨边界时直接被截断。

推荐：`doc_stride = int(max_paragraph_len × 0.25)`（75% 重叠）

知乎文章在大模型上测试：×0.25 → Private 0.831，×0.1 → 0.843。

### 2. 随机窗口偏移防止捷径学习

原始代码始终将答案放在窗口正中，模型会学到"答案就在中间"。

加随机偏移：`offset = np.random.randint(-max_paragraph_len//2, max_paragraph_len//2)`

| 偏移范围 | Private Score |
|---|---|
| 无偏移 | 0.701 |
| max_len // 4 | 0.726 |
| **max_len // 2** | **0.732** |
| max_len | 0.728（偏移太大窗口可能不含答案） |

## 三、后处理

### 1. Logit ≠ 概率

`output.start_logits` 是未经 softmax 的原始分数，直接相加无意义。

正确做法：先 `torch.softmax(logits, dim=0)` 转概率，再相乘（或 log_softmax 后相加）。

### 2. start_index 必须 ≤ end_index

不加校验时，模型可能选出 `start > end` 的无效答案，导致空字符串。

```python
if start_index <= end_index:
    prob = start_prob * end_prob
    ...
else:
    continue
```

这个修复在 Boss baseline 中贡献了约 0.01 的分数提升。

## 四、模型选择

| 模型 | 参数量 | Private Score（知乎文章） |
|---|---|---|
| bert-base-chinese | 103M | 0.732 |
| bert-base-multilingual-cased | 179M | 0.775 |
| chinese-macbert-large | 324M | 0.831 |
| **luhua/chinese_pretrain_mrc_macbert_large** | **324M** | **0.842** |

MRC（Machine Reading Comprehension）专用预训练模型比通用 BERT 在 QA 任务上强很多。

大模型在 T4 15GB 上需要：batch_size=4 + gradient_accumulation_steps=8 + fp16。

## 五、训练策略

### 1. Early stopping 是必备手段

过拟合后模型表现会急剧下降，只保存最后一个 epoch 的权重会丢失最优结果。

每轮验证后比较，`dev_acc > best_acc` 时才保存。

### 2. Epoch 数不是越多越好

大模型在 2~3 epoch 就能充分收敛，跑太多反而过拟合。

知乎文章：epoch=16 的训练集 acc 比 epoch=1 高 0.073，但 Kaggle 分数反而更低。

### 3. 训练 acc 趋势是健康指标

- 正常：acc 随 step 稳步上升或持平
- 异常：acc 随 step 持续下降 → lr 过高，模型在过冲

## 六、参考资源

- 知乎文章：https://zhuanlan.zhihu.com/p/721275772
- CSDN 实验记录：https://blog.csdn.net/qq_42994201/article/details/121442992
- 博客园 Strong Baseline 解析：https://www.cnblogs.com/SkyRainWind/p/18017978
