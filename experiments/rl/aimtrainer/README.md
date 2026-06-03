# Aim Trainer — 强化学习瞄准仿真环境

基于 Gymnasium + Stable-Baselines3 的多目标瞄准训练环境，配套 HTML 游戏和 Pygame 可视化。

## 快速体验

### 🌐 浏览器游戏（人类可玩）

直接打开 `src/aim_trainer_game.html`，用鼠标瞄准点击。支持两种模式：点击射击 / 瞄准即命中。

### 🤖 AI 推理演示（训练好的模型在浏览器中自动瞄准）

```bash
uv run python experiments/rl/aimtrainer/src/ai_play.py
```

打开后点击「🤖 AI 演示」按钮，Python 端加载训练好的 PPO 模型，实时推理并通过 eel 推送状态到浏览器渲染。准星自动移动、锁定目标、命中射击。

### 🎮 Pygame 人类对战

```bash
uv run python experiments/rl/aimtrainer/src/human_play.py
```

鼠标点击射击，空格重置，P 暂停。

### 👁️ 训练直播（边训边看 agent 表现）

```bash
uv run python experiments/rl/aimtrainer/src/train_visual.py --steps 50000 --render-freq 5000
```

每 5000 步暂停训练，弹出窗口展示 agent 当前水平。

### 📊 纯训练（无窗口，更快）

```bash
uv run python experiments/rl/aimtrainer/src/train.py --steps 200000
```

### 📈 评估模型

```bash
uv run python experiments/rl/aimtrainer/src/eval.py --episodes 10
```

### 🧪 运行测试

```bash
uv run --group dev pytest experiments/rl/aimtrainer/tests/ -v
```

## 文件结构

```
aimtrainer/
  env/
    aim_trainer_env.py    # Gymnasium 环境核心
  src/
    aim_trainer_game.html # 🌐 浏览器瞄准游戏（人类 + AI 双模式）
    ai_play.py            # 🤖 AI 推理演示（模型驱动浏览器）
    train.py              # 📊 SB3 PPO 纯训练
    train_visual.py       # 👁️ 训练 + 实时渲染
    human_play.py         # 🎮 Pygame 人类对战
    eval.py               # 📈 评估 + GIF 录制
  tests/
    test_env.py           # 🧪 环境单元测试
  artifacts/              # 模型产物（gitignore）
```

## 环境设计

| 项目 | 内容 |
|---|---|
| 状态 | `[crosshair_xy(2), target1_xy_alive(3), ...]` |
| 动作 | `[dx, dy]` 连续位移 |
| 奖励 | +10 命中 / -0.01 步长 / +距离辅助 |
| 目标 | 5 个红圈同时存在，命中后刷新 |
| 算法 | PPO (Stable-Baselines3) |

## 依赖

`pyproject.toml` 中已包含：

- `gymnasium` — 强化学习环境框架
- `stable-baselines3` — PPO 算法实现
- `pygame` — 可视化渲染
- `numpy` — 数值计算
- `matplotlib` — 曲线绘制
