# HW11 DANN 实验结果记录

> 更新日期：2026-07-09
> 代码入口：`experiments/ML2021-Spring/HW11/npu/hw11_npu.py`
> 任务：单源域适应（Real → Drawing 手写数字分类）

## 成绩记录

| 分数 | 榜单 | Epochs | 备注 |
|:---|---|:---:|---|
| **0.73274** | Public | 200 | 基础款 DANN，无 TTA / Entropy / 伪标签 |
| **0.73112** | Private | 200 | 同上，同一次提交 |
| 0.71882 | Public | 100 | 旧版（含 Entropy + 伪标签），epoch 减半后分数下降 |
| 0.72050 | Private | 100 | 同上 |

## 结论

**基础款 DANN（ResNet18 + adaptive_lambda + cosine LR）+ loss_balance + Decoder AE + margin 伪标签** 已达 strong baseline（0.73274 / 0.73112）。

未做：TTA、Entropy Minimization、EMA Teacher、Ensemble。不考虑 boss baseline。到此为止。
