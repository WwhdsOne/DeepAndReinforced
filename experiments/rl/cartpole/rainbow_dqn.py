"""
Rainbow DQN — 纯 PyTorch 实现。

与 basic_dqn.py 的对比：

  basic_dqn.py                →  rainbow_dqn.py
  ──────────────────────────────────────────────────
  QNetwork (nn.Linear)        →  NoisyLinear + Dueling
  ReplayBuffer (list)         →  PrioritizedReplayBuffer (IS 加权)
  epsilon-greedy 探索           →  NoisyNet 参数噪声（自动）
  MSE Loss (标量 Q)            →  C51 分布学习（CrossEntropy）
  单步 TD 目标                  →  n_step=3 多步累计
  手写 for ep 循环              →  保留手写循环（可读性强）

Rainbow = DQN + Double + Dueling + NoisyNet + C51 + PER + Multi-step

使用方法：
  uv run python experiments/rl/cartpole/rainbow_dqn.py
"""

from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# ═══════════════════════════════════════════════════
# 网络：Dueling + NoisyLinear + C51 分布输出
# ═══════════════════════════════════════════════════

class RainbowNet(nn.Module):
    """Rainbow 网络：共享特征 + Dueling 分支 + NoisyLinear + 分布输出。

    输出形状: (batch, action_dim, num_atoms) — 每个动作一个分布。
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        # 支撑集 z_i = v_min + i * dz
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        # 共享特征
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value 流 → num_atoms
        self.value = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, num_atoms),
        )

        # Advantage 流 → action_dim * num_atoms
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * num_atoms),
        )

        # 初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 logits: (batch, action_dim, num_atoms)"""
        feat = self.features(x)
        v = self.value(feat)                       # (B, num_atoms)
        a = self.advantage(feat).view(-1, self.action_dim, self.num_atoms)
        # Dueling: Q = V + A - mean(A)
        return v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """从分布 logits 计算期望 Q 值: (batch, action_dim)"""
        logits = self(x)
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.support).sum(dim=-1)

    def reset_noise(self):
        """重置所有 NoisyLinear 的噪声参数。"""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ═══════════════════════════════════════════════════
# NoisyLinear：带参数噪声的全连接层
# ═══════════════════════════════════════════════════

class NoisyLinear(nn.Module):
    """Noisy Linear 层：weight = μ_w + σ_w ⊙ ε_w, bias = μ_b + σ_b ⊙ ε_b。

    训练时噪声自动重采样，测试时关闭（使用 μ）。
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 可学习参数
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 注册噪声缓冲区
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            mu_range * 0.5 / np.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(mu_range * 0.5 / np.sqrt(self.in_features))

    def reset_noise(self):
        """采样因子化高斯噪声。"""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


# ═══════════════════════════════════════════════════
# 优先经验回放（PER）
# ═══════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """优先经验回放 + n 步累计奖励存储。

    使用 SumTree 实现 O(log N) 采样。
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = np.empty(capacity, dtype=object)
        self.pos = 0
        self.size = 0

        # SumTree 数组（2 * capacity 叶子二叉树）
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        self.tree = np.zeros(2 * self.tree_size, dtype=np.float32)

        # n 步累计缓存
        self.n_step_buffer: deque = deque(maxlen=3)
        self.gamma = 0.99

    def push(self, state, action, reward, next_state, done, td_error=None):
        """存入经验，优先用 n 步累计奖励。"""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) >= 3:
            state0, action0, _, _, _ = self.n_step_buffer[0]
            _, _, _, next_state_n, done_n = self.n_step_buffer[-1]

            # n 步累计奖励
            n_reward = 0.0
            for k in range(3):
                _, _, r, _, d = self.n_step_buffer[k]
                n_reward += (self.gamma ** k) * r
            if not done_n:
                n_reward += 0  # bootstrapping 在训练时处理

            self._store(state0, action0, n_reward, next_state_n, done_n, td_error)

    def _store(self, state, action, reward, next_state, done, td_error):
        priority = (abs(td_error) ** self.alpha + 1e-6) if td_error is not None else 1.0
        self.buffer[self.pos] = (state, action, reward, next_state, done, priority)

        # 更新 SumTree
        tree_idx = self.tree_size + self.pos
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """按优先级采样，返回 (batch, indices, IS_weights)。"""
        segment = self.tree[1] / batch_size
        indices = np.zeros(batch_size, dtype=int)
        weights = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = random.uniform(a, b)

            # 从 SumTree 中采样
            tree_idx = 1
            while tree_idx < self.tree_size:
                left = 2 * tree_idx
                if value <= self.tree[left]:
                    tree_idx = left
                else:
                    value -= self.tree[left]
                    tree_idx = left + 1

            buf_idx = tree_idx - self.tree_size
            indices[i] = buf_idx

            # IS 权重
            prob = self.tree[tree_idx] / max(self.tree[1], 1e-6)
            weights[i] = (self.size * prob) ** (-self.beta)

        # 归一化 IS 权重
        weights /= max(weights.max(), 1e-6)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 组装 batch（经验已存为 tensor，直接 stack）
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        states = torch.stack(list(states))
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(list(next_states))
        dones = torch.tensor(dones, dtype=torch.bool)

        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新采样经验的新优先级。"""
        priorities = (np.abs(td_errors) ** self.alpha + 1e-6)
        for idx, pri in zip(indices, priorities):
            tree_idx = self.tree_size + idx
            self.tree[tree_idx] = pri
            while tree_idx > 1:
                tree_idx //= 2
                self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def preprocess_state(state) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.float()
    return torch.tensor(state, dtype=torch.float32)


# ═══════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════

def train(episodes: int = 5_000):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 两个网络：online (Noisy) 和 target (Noisy but eval mode → 用 μ)
    online_net = RainbowNet(state_dim=state_dim, action_dim=action_dim)
    target_net = RainbowNet(state_dim=state_dim, action_dim=action_dim)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=5e-4)
    replay_buffer = PrioritizedReplayBuffer(capacity=100_000, alpha=0.6, beta=0.4)

    gamma = 0.99
    batch_size = 64
    target_update_freq = 100

    for ep in range(episodes):
        online_net.train()
        online_net.reset_noise()
        target_net.reset_noise()

        state, _ = env.reset()
        total_reward = 0
        done = False
        td_errors = []

        while not done:
            # 动作选择（NoisyNet 训练模式自动探索）
            with torch.no_grad():
                state_tensor = preprocess_state(state).unsqueeze(0)
                q = online_net.q_values(state_tensor)
                action = q.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 使用原始 reward（不再加手工惩罚，C51 分布学习自行拟合）
            replay_buffer.push(
                preprocess_state(state),
                action,
                reward,
                preprocess_state(next_state),
                done,
            )

            state = next_state
            total_reward += reward

            # 经验回放
            if len(replay_buffer) >= batch_size:
                (states, actions, rewards, next_states, dones,
                 indices, is_weights) = replay_buffer.sample(batch_size)

                # ── 计算 TD 目标分布（Double DQN） ──
                online_net.eval()
                with torch.no_grad():
                    # Double: 用 online 选动作，target 算分布
                    next_q_online = online_net.q_values(next_states)
                    next_actions = next_q_online.argmax(dim=1)  # (B,)

                    next_logits = target_net(next_states)  # (B, A, N)
                    next_logits = next_logits[range(batch_size), next_actions]  # (B, N)
                    next_probs = torch.softmax(next_logits, dim=-1)

                    # C51 投影：Tz_j = r + γ · z_j · (1 - d)
                    gamma_z = gamma * online_net.support.unsqueeze(0)  # (1, N)
                    Tz = rewards.unsqueeze(1) + gamma_z * (~dones).unsqueeze(1).float()
                    Tz = Tz.clamp(online_net.support[0].item(), online_net.support[-1].item())

                    # 投影到支撑集
                    b = (Tz - online_net.support[0].item()) / (
                        online_net.support[1] - online_net.support[0]
                    )
                    l = b.floor().long()
                    u = b.ceil().long()
                    # 矢量化的 C51 投影（替代双层循环，快 100x+）
                    l_clamped = l.clamp(0, online_net.num_atoms - 1)
                    u_clamped = u.clamp(0, online_net.num_atoms - 1)

                    l_weight = next_probs * (u - b)     # 下溢部分权重
                    u_weight = next_probs * (b - l)     # 上溢部分权重

                    target_dist = torch.zeros(batch_size, online_net.num_atoms)
                    target_dist.scatter_add_(1, l_clamped, l_weight)
                    target_dist.scatter_add_(1, u_clamped, u_weight)
                online_net.train()

                # ── 损失：KL 散度（C51 分布损失） ──
                current_logits = online_net(states)  # (B, A, N)
                current_logits = current_logits[range(batch_size), actions]  # (B, N)
                current_log_probs = torch.log_softmax(current_logits, dim=-1)

                loss = -(target_dist * current_log_probs).sum(dim=-1)
                loss = (loss * is_weights).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                # 更新优先级（KL 散度作为 TD 误差）
                with torch.no_grad():
                    td_error = loss.detach().cpu().numpy().flatten()
                    if len(td_error) == 1:
                        td_error = np.full(batch_size, td_error[0])
                replay_buffer.update_priorities(indices, td_error)

        # 更新 target 网络
        if ep % target_update_freq == 0 and ep > 0:
            target_net.load_state_dict(online_net.state_dict())
            target_net.eval()
            print(f"Episode {ep:5d} | Reward = {total_reward:6.1f}")

            # 评估一局
            if ep % 100 == 0:
                _eval(online_net)

    env.close()
    print("\n训练完成。")


def _eval(net: RainbowNet):
    """评估一局（关闭噪声，贪心策略）。"""
    net.eval()
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    total = 0
    done = False
    with torch.no_grad():
        while not done:
            env.render()
            q = net.q_values(preprocess_state(state).unsqueeze(0))
            action = q.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    env.close()
    print(f"  >>> 评估回合: {total:.1f}")
    net.train()


if __name__ == "__main__":
    train()
