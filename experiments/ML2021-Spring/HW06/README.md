# HW06 - 生成对抗网络 (GAN)

动漫人脸生成，基于 DCGAN + WGAN-GP。

## WGAN-GP 损失函数

**判别器 D**（最小化）：

$$\mathcal{L}_D = \underbrace{-\mathbb{E}_{x \sim p_r}[D(x)] + \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})]}_{\text{Wasserstein 距离估计}} + \underbrace{\lambda \cdot \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} \left[ \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2 \right]}_{\text{梯度惩罚 (GP)}}$$

**生成器 G**（最小化）：

$$\mathcal{L}_G = -\mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})]$$

**插值采样**：

$$\hat{x} = \alpha x + (1 - \alpha) \tilde{x}, \quad \alpha \sim U[0, 1]$$

| 符号 | 含义 |
|------|------|
| $x$ | 真实图像，$x \sim p_r$ |
| $\tilde{x}$ | 生成图像，$\tilde{x} = G(z),\ z \sim \mathcal{N}(0, I)$ |
| $\hat{x}$ | 真实与生成的插值 |
| $\lambda$ | 梯度惩罚系数（代码中取 10） |
| $n_{\text{critic}}$ | 每训练 D 的次数，训练一次 G（代码中取 5） |

## 经验教训

### 梯度惩罚 `.mean()` 导致 GP 完全失效

**现象**：Loss_D 绝对值巨大（-300 到 -1000），Critic 无约束地拉大真假分差。

**根因**：梯度惩罚函数中用 `outputs=d_interpolates.mean()` 调用 `autograd.grad`。`D(interpolates)` 输出形状 `(N,)`，每个 `D(x_hat_i)` 只依赖 `x_hat_i`。`.mean()` 等价于 `sum() / N`，使 `autograd.grad` 对每个样本的梯度缩小 N 倍：

- 梯度范数缩小 N 倍
- `((norm - 1) ** 2)` 中 norm ≈ 0，GP ≈ 1
- 但有效惩罚强度 ≈ λ / N² = 10 / 4096² ≈ 6×10⁻⁷
- 等于完全没有 Lipschitz 约束

**修复**：用 `grad_outputs=torch.ones_like(d_interpolates)` 替代 `.mean()`，确保每个样本获得正确的逐样本梯度：

```python
gradients = torch.autograd.grad(
    outputs=d_interpolates,
    inputs=interpolates,
    grad_outputs=torch.ones_like(d_interpolates),
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
)[0]
gradients = gradients.view(batch_size, -1)
gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
```

**结论**：`autograd.grad` 的 `outputs` 参数决定了梯度的缩放。对向量输出（如判别器的逐样本打分），必须用 `grad_outputs` 指定正确的上流梯度，而非先 `.mean()` 再求导。

### FP16 + WGAN-GP 梯度惩罚失效

**现象**：Loss_D 从 ~200 持续上涨到 ~237，Loss_G 同步下降，梯度惩罚未能约束判别器。

**根因**：`gradient_penalty` 中 `create_graph=True` 需要计算二阶梯度（对梯度再求导）。在 FP16 下：
- FP16 仅 5 位指数、10 位尾数，二阶梯度精度严重不足
- 小值梯度直接下溢为 0，大值梯度溢出为 `inf`
- 计算出的梯度范数不准，GP 惩罚项形同虚设
- D 失去 Lipschitz 约束，`D(real) - D(fake)` 无限拉大

**修复**：直接移除 FP16，全部使用原生 FP32。WGAN-GP 的二阶梯度对精度敏感，FP16 带来的速度提升不值得引入的数值风险。

**结论**：混合精度训练中，涉及高阶导数的计算（梯度惩罚、Hessian、meta-learning 等）必须保持 FP32。对于 GAN 这类训练不稳定、调参成本高的场景，直接用 FP32 更省心。

### Discriminator (Critic) 不应使用 BatchNorm

**问题**：代码中 `Discriminator` 的 `conv_bn_lrelu` 使用了 `BatchNorm2d`，这直接来自 DCGAN 论文。但 WGAN-GP 论文中明确指出 **Critic 不应使用 BatchNorm**。

**原因**：BatchNorm 对 batch 做归一化（减均值除方差），这本身就是一个依赖于 batch 统计量的非线性操作，会破坏 Critic 的 Lipschitz 连续性。GP 约束的是 $\|\nabla_{\hat{x}} D(\hat{x})\|_2 \approx 1$，但如果 BN 改变了输入的 Lipschitz 常数，GP 的约束就不再可靠，表现为：
- 训练初期看似稳定，后期 Loss 震荡或突然崩溃
- 梯度惩罚数值正常但 Lipschitz 约束实际未生效

**修复**：将 Critic 的 `conv_bn_lrelu` 改为不带 BN 的版本：

```python
def conv_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        nn.LeakyReLU(0.2),
    )
```

> **注意**：Generator 中的 BatchNorm 可以保留，DCGAN 的 BN + ReLU 组合对生成器仍然有效。只有 Critic 需要去掉 BN。

### GAN 的 batch_size 不是越大越好

**现象**：batch_size=4096 训练到 70-80 epoch 后 Loss 不再变化，生成质量停滞。

**根因**：GAN 与分类任务对 batch_size 的需求相反：

| | 分类任务 | GAN |
|--|---------|-----|
| 关注点 | 每个 step 的梯度质量 | step 数量（更新频率） |
| 大 batch | 更稳的梯度 → 更快收敛 | 每 epoch step 太少 → D/G 互相"看不动"对方 |
| 小 batch | 梯度噪声大 → 收敛慢 | 更多更新 → D/G 动态博弈更充分 |

batch_size=4096 时，~70k 数据集每 epoch 仅 ~17 步，D 和 G 的更新次数严重不足。

**修复**：将 batch_size 从 4096 降至 1024（每 epoch ~68 步），兼顾 GPU 利用率和更新频率。

**结论**：GAN 训练中 step 数量比单步效率更重要。GPU 开销主要是 per-step 的固定部分，小 batch 单步更快，总 wall time 差距不大，但模型学到的东西多得多。

## 参考论文

| 论文 | 代码对应 |
|------|----------|
| Goodfellow et al. (2014). [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661). NeurIPS 2014. | GAN 基本框架：Generator 与 Discriminator 对抗训练 |
| Radford et al. (2016). [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). ICLR 2016. | DCGAN 架构：`Generator`（转置卷积 + BN + ReLU）、`Discriminator`（步幅卷积 + BN + LeakyReLU）、`weights_init` 权重初始化 |
| Arjovsky et al. (2017). [Wasserstein GAN](https://arxiv.org/abs/1701.07875). ICML 2017. | Wasserstein 损失函数（无 Sigmoid）、`n_critic=5`、RMSprop 优化器（后被 WGAN-GP 替代） |
| Gulrajani et al. (2017). [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028). NeurIPS 2017. | `gradient_penalty` 梯度惩罚（替代权重裁剪）、`Adam(betas=(0.0, 0.9))` 优化器、$\lambda=10$ |
