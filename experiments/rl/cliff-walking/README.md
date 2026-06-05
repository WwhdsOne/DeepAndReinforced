# 悬崖漫步（CliffWalking）

4×12 网格世界的强化学习环境，纯 NumPy 实现。

| 符号 | 含义 |
|------|------|
| `A` | Agent 当前位置 |
| `C` | 悬崖（掉入 → -100，回到起点） |
| `S` | 起点 (3, 0) |
| `G` | 终点 (3, 11) |
| `.` | 普通格子（每步 -1） |

## 命令

```bash
# 键盘试玩（W=上 D=右 S=下 A=左 Q=退出 R=重置）
uv run cliff-walking-play

# 训练（默认 500 轮）
uv run cliff-walking-train --episodes 500

# DQN 训练（神经网络逼近 Q 函数）
uv run cliff-walking-dqn --episodes 500

# 运行测试
uv run --group dev pytest experiments/rl/cliff-walking/tests/ -v
```

## 使用示例

```python
from cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()
state = env.reset()                  # 状态 36，即 (3, 0)
state, reward, done = env.step(1)    # 向右走
env.render()                         # 终端打印网格
```
