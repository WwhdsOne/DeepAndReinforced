import argparse
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch
import random
import numpy as np           # ✅ 放到最上面

"""
属性	描述
目标	通过左右移动小车，保持杆子直立不倒。
观测状态	一个包含4个连续值的数组：[小车位置, 小车速度, 杆子角度, 杆子角速度]。
动作空间	0 或 1（代表将小车向左或向右推）。
奖励机制	杆子每保持直立 1 秒，就获得 +1 分。
终止条件	杆子倾斜角度过大（超过约12度）或小车移出边界（超过±2.4单位）。
成功标准	在100个连续回合中，平均得分达到 195 分，通常就认为问题已解决。
"""


class QNetwork(nn.Module):
    """简单的全连接神经网络，输入是状态，输出是每个动作的 Q 值。"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):  # ✅ 添加 done
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)  # ✅ 解包 dones

        # 转换为张量
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)  # ✅ 转换为布尔张量

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
def position_penalty(state, penalty_coef=0.1):
    """
    根据小车偏离中心的距离计算惩罚值。
    状态格式: [x, x_dot, theta, theta_dot]
    x 的范围约为 [-2.4, 2.4]
    """
    x = state[0] if isinstance(state, (list, np.ndarray)) else state[0].item()
    return penalty_coef * (x ** 2)


def preprocess_state(state):
    """
    将 CartPole 的连续状态转换为张量

    Args:
        state: numpy array 或 list，形状 (4,)

    Returns:
        torch.Tensor，形状 (4,)
    """
    if isinstance(state, torch.Tensor):
        return state
    elif isinstance(state, np.ndarray):  # ✅ 导入 numpy 处理
        return torch.from_numpy(state).float()
    else:
        return torch.tensor(state, dtype=torch.float32)

def angle_penalty(state, coef=0.5):
    theta = state[2] if isinstance(state, (list, np.ndarray)) else state[2].item()
    return coef * (theta ** 2)

def evaluate(weights: str):
    """加载权重并展示一次效果（无步数上限）。"""
    network = QNetwork(state_dim=4, action_dim=2)
    network.load_state_dict(torch.load(weights, weights_only=True))
    network.eval()
    print(f"已加载权重文件: {weights}")

    env = gym.make("CartPole-v1", render_mode="human")
    env._max_episode_steps = float("inf")
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = preprocess_state(state).unsqueeze(0)
        with torch.no_grad():
            action = network(state_tensor).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    env.close()
    print(f"评估回合奖励: {total_reward:.2f}")
    return total_reward


def train(episodes: int = 5000, weights: str = None):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env._max_episode_steps = float("inf")
    q_network = QNetwork(state_dim=4, action_dim=2)
    target_network = QNetwork(state_dim=4, action_dim=2)

    if weights:
        q_network.load_state_dict(torch.load(weights, weights_only=True))
        print(f"已加载权重文件: {weights}")

    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=5e-4)
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.01
    gamma = 0.99
    batch_size = 64
    target_update_freq = 100
    best_weights_appear = False
    best_eval_reward = -float('inf')

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = preprocess_state(state).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            penalty = position_penalty(state) + angle_penalty(state)
            if done and truncated:
                boundary_penalty = -100.0
            else:
                boundary_penalty = 0.0
            reward = reward - penalty - boundary_penalty

            replay_buffer.push(
                preprocess_state(state),
                action,
                reward,
                preprocess_state(next_state),
                done
            )

            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size and not best_weights_appear:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = target_network(next_states).max(1)[0]
                    target_q = rewards + gamma * next_q * (~dones)

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if ep % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
            print(f"Episode {ep}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

            # 保存临时权重用于评估
            tmp_path = "cartpole_dqn_tmp.pth"
            torch.save(target_network.state_dict(), tmp_path)
            eval_reward = evaluate(tmp_path)

            if eval_reward >= 300 and eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                q_network_path = f"cartpole_dqn_episode_{ep}.pth"
                torch.save(q_network.state_dict(), q_network_path)
                best_weights_appear = True
                print(f"  >>> 已保存最佳模型权重到 {q_network_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole DQN")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--weights", type=str, default=None, help="预训练权重文件路径 (.pth)")
    train_parser.add_argument("--episodes", type=int, default=5000, help="训练回合数")

    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("weights", type=str, help="权重文件路径 (.pth)")

    args = parser.parse_args()

    if args.command == "train":
        train(episodes=args.episodes, weights=args.weights)
    elif args.command == "eval":
        evaluate(args.weights)
    else:
        parser.print_help()