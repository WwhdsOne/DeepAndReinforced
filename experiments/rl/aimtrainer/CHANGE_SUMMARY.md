# AimTrainer 变更总结

本文档总结本次从初版到当前版本对 `experiments/rl/aimtrainer/` 的主要改动。

## 1. 环境奖励与控制逻辑

初版问题：
- `target_radius` 变小时，命中区域、靠近奖励区域、默认步长一起缩小，训练容易变稀疏。
- agent 容易在目标边缘左右震荡，或者在中距离区域来回试探。
- 到达边界后，即使实际位置没有移动，只要动作幅度不小，就不会被当成无效动作处罚。

当前改动：
- 默认 `max_steps` 从较低值提升到更高步数，允许更细粒度控制。
- 默认 `action_step` 改为更小的缩放逻辑：
  - 当前默认是 `max(target_radius * 0.4, 0.004)`
- `near_zone` 不再严格等于命中半径，而是保留更大的 shaping 区域：
  - `max(target_radius * near_zone_scale, near_zone_min)`
- 增加连续距离变化奖励：
  - `reward_distance_delta`
  - 朝最近目标更近时给正反馈
  - 远离时给负反馈
- 增加历史最优进展奖励：
  - `reward_progress`
  - 只有刷新历史最近距离，或者首次进入引导区时才会增加
- 增加远离目标的额外惩罚倍率：
  - 当前参数名：`regress_penalty_scale`
  - 作用：让“靠近一点再退回去”的来回抖动净收益变差
- `idle_penalty` 改为根据“实际位移”而不是“原始动作”计算：
  - 顶墙、卡边、越界无效动作现在会被视为无效移动并受罚

## 2. 观测空间增强

初版问题：
- reward 按“最近目标”计算
- 但 observation 只平铺所有目标坐标，没有显式告诉 agent “当前最近目标在什么方向”
- 这容易导致开局固定方向偏移、击中后切目标不稳定、在局部区域乱晃

当前改动：
- observation 末尾新增两个维度：
  - `nearest_target_dx`
  - `nearest_target_dy`
- 观测维度从：
  - `2 + n_targets * 3`
  变为：
  - `2 + n_targets * 3 + 2`
- `observation_space` 范围同步从纯 `[0, 1]` 改为允许相对向量的 `[-1, 1]`

影响：
- 新模型能够直接利用“最近目标相对准星的方向”
- 旧模型因为观测维度变化，不能直接复用继续训练

## 3. 可视化训练参数调整

初版问题：
- 可视化训练脚本中的 reward 配置偏激进
- 某些参数组合容易导致 agent 远处乱晃、边缘顶墙、局部来回抖动

当前 `train_visual.py` 中使用的环境配置更偏向稳定训练：
- `max_steps=800`
- `hit_reward=120.0`
- `time_penalty=0.05`
- `progress_coef=0.25`
- `distance_delta_coef=1.0`
- `efficiency_coef=3.0`
- `idle_threshold=0.01`
- `time_ramp=4.0`

## 4. 可视化 HUD 信息

初版 HUD：
- 只显示 `Step / Hits / Reward`

中间版本：
- 增加过固定参数面板，例如：
  - `Target Radius`
  - `Action Step`
  - `Hit Reward`
  - `Time Penalty`
  等

当前版本：
- 改为实时显示本步 reward 分解，不再强调固定配置项
- 左上角实时展示：
  - `Hit Reward`
  - `Progress Reward`
  - `Delta Reward`
  - `Time Penalty`
  - `Idle Penalty`

这些值来自环境 `step()` 返回的 `info` 字段中的 reward breakdown。

## 5. 可视化展示步数

初版问题：
- 可视化 demo 采用独立的固定展示步数，和环境当前 `max_steps` 不一致

当前改动：
- 如果没有显式指定展示上限，则 demo 默认使用当前环境的 `max_steps`
- 这样训练回合步数和本次可视化展示步数保持一致

## 6. 训练入口与参数演化

当前训练使用的核心算法与优化配置如下：

- 强化学习算法：`PPO`
- 策略网络：`MlpPolicy`
- 神经网络优化器：`Adam`（Stable-Baselines3 默认）
- `n_steps=2048`
- `batch_size=64`
- `n_epochs=10`
- `learning_rate=3e-4`
- `ent_coef=0.1`
- `device="auto"`

环境包装与归一化：

- `DummyVecEnv`
- `VecNormalize`
  - `norm_obs=True`
  - `norm_reward=True`
  - `clip_obs=10.0`
  - `clip_reward=10.0`

本轮过程中曾短暂加入：
- `--max-steps`
- `--action-step`

后来按要求已回退。

当前保留的命令行参数：

`train.py`
- `--steps`
- `--eval-freq`
- `--target-radius`

`train_visual.py`
- `--steps`
- `--render-freq`
- `--target-radius`

## 7. 模型保存与继续训练

当前训练脚本会保存：
- `ppo_aim_trainer.zip`
- `vec_normalize_stats.pkl`

但不会自动继续训练：
- 下次启动脚本不会自动 `load()` 上次模型
- 也不会自动恢复上次的 `VecNormalize` 统计量
- 当前默认行为仍然是重新初始化训练

## 8. 测试补充

本轮为环境补充并更新了多项测试，覆盖：
- 默认步长/步数缩放
- 连续距离奖励
- 远离目标的更重惩罚
- 顶墙无效动作惩罚
- reward breakdown 是否写入 `info`
- observation 中是否包含最近目标相对向量

当前环境测试状态：
- `experiments/rl/aimtrainer/tests/test_env.py`
- 已通过：`15 passed`

## 9. 当前版本的核心思路

当前版本的方向可以概括为：
- 用更高 `max_steps` + 更小 `action_step` 提升局部控制精度
- 用连续距离奖励引导稳定逼近
- 对远离目标和无效动作加更明确的惩罚
- 在 observation 中显式给出最近目标方向
- 在可视化 HUD 中直接展示实时 reward 分解，便于诊断训练行为

## 10. 当前仍需关注的问题

虽然本轮已经显著增强了环境和诊断能力，但以下问题仍值得继续观察：
- agent 是否仍会在某些中距离区域来回振荡
- 最近目标切换后是否仍会出现短时失稳
- `pygame` 与 `cv2` 的 SDL 重复加载警告仍存在，可能导致可视化运行时的潜在不稳定
