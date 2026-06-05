import torch.nn as nn
import torch
import argparse
import torch.optim as optim
import random
from env import CliffWalkingEnv

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        n_states = env.n_states  # 48
        n_actions = env.n_actions  # 4
        
        self.network = nn.Sequential(
            nn.Linear(n_states, 128),  # 输入：48 维 one-hot 状态
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)  # 输出：4 个动作的 Q 值
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))

    def __len__(self):
        return len(self.buffer)


def train(env: CliffWalkingEnv, episodes: int = 500) -> torch.Tensor:
    """训练主函数 —— 在此实现你的算法。

    Parameters
    ----------
    env : CliffWalkingEnv
        悬崖漫步环境。
    episodes : int
        训练轮数。

    Returns
    -------
    Q : torch.Tensor
        训练后的 Q 表，形状 (48, 4)。
    """
    replay_buffer = ReplayBuffer()
    q_network = QNetwork(env)
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    target_network = QNetwork(env)
    target_network.load_state_dict(q_network.state_dict()) # 同步目标网络（完全复制）
    n_states = env.n_states
    n_actions = env.n_actions

    Q = torch.zeros((n_states, n_actions))

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if torch.rand(1).item() < 0.1:
                action = torch.randint(0, n_actions, (1,)).item()
            else:
                action = Q[state].argmax()

            next_state, reward, done = env.step(action)

            Q[state, action] = Q[state, action] + 0.1 * (
                reward + 0.9 * Q[next_state].max() - Q[state, action]
            )

            state = next_state

    return Q


def evaluate(env: CliffWalkingEnv, Q: torch.Tensor, episodes: int = 10) -> float:
    """评估训练后的策略。

    Parameters
    ----------
    env : CliffWalkingEnv
    Q : torch.Tensor
        Q 表，形状 (48, 4)。
    episodes : int
        评估轮数。

    Returns
    -------
    avg_reward : float
        平均累积奖励。
    """
    total = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = Q[state].argmax()  # 贪婪策略
            state, reward, done = env.step(action)
            total += reward
    return total / episodes


def render_policy(env: CliffWalkingEnv, Q: torch.Tensor) -> None:
    """在终端渲染学到的策略轨迹。"""
    action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    state = env.reset()
    env.render()
    done = False
    steps = 0
    while not done and steps < 100:
        action = Q[state].argmax().item()
        print(f"动作: {action_names[action]}")
        state, reward, done = env.step(action)
        env.render()
        steps += 1
        if done:
            print("🎉 到达终点！" if state == 47 else "💀 掉下悬崖！")


def main() -> None:
    parser = argparse.ArgumentParser(description="悬崖漫步 — 强化学习训练")
    parser.add_argument("--episodes", type=int, default=500, help="训练轮数")
    parser.add_argument("--eval", action="store_true", help="仅评估（需已有 Q 表）")
    args = parser.parse_args()

    env = CliffWalkingEnv()

    if args.eval:
        # 训练 + 评估
        Q = train(env, episodes=args.episodes)
        avg = evaluate(env, Q, episodes=10)
        print(f"\n平均奖励: {avg:.2f}")
        render_policy(env, Q)
    else:
        Q = train(env, episodes=args.episodes)
        avg = evaluate(env, Q, episodes=10)
        print(f"\n训练完成 → 平均奖励: {avg:.2f}")
        render_policy(env, Q)


if __name__ == "__main__":
    main()
