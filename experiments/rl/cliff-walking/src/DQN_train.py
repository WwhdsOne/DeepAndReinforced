import torch.nn as nn
import torch
import argparse
import torch.optim as optim
import random
import numpy as np
from env import CliffWalkingEnv

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        n_states = env.n_states  # 48
        n_actions = env.n_actions  # 4
        
        self.network = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
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
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.stack(states)  # (batch_size, 48)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def state_to_onehot(state, n_states=48):
    """将状态索引转换为 one-hot 向量"""
    one_hot = torch.zeros(n_states)
    one_hot[state] = 1.0
    return one_hot

def train(env: CliffWalkingEnv, episodes: int = 500):
    replay_buffer = ReplayBuffer(capacity=10000)
    q_network = QNetwork(env)
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    target_network = QNetwork(env)
    target_network.load_state_dict(q_network.state_dict())
    n_states = env.n_states
    n_actions = env.n_actions

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    batch_size = 64
    target_update_freq = 10  # 更频繁地更新目标网络
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # ε-greedy 策略
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    state_tensor = state_to_onehot(state, n_states).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            
            # 存储经验
            replay_buffer.push(
                state_to_onehot(state, n_states),
                action,
                reward,
                state_to_onehot(next_state, n_states),
                done
            )
            
            state = next_state
            total_reward += reward
            
            # 经验回放
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 当前 Q 值
                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                # 目标 Q 值
                with torch.no_grad():
                    next_q = target_network(next_states).max(1)[0]
                    target_q = rewards + gamma * next_q * (~dones)  # done 时 next_q 为 0
                
                # 更新网络
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 衰减 epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # 定期更新目标网络
        if (ep + 1) % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # 打印进度
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    
    return q_network

def evaluate(env: CliffWalkingEnv, q_network: QNetwork, episodes: int = 10) -> float:
    """评估训练后的策略。
    
    Parameters
    ----------
    env : CliffWalkingEnv
    q_network : QNetwork
        训练好的 Q 网络。
    episodes : int
        评估轮数。
    
    Returns
    -------
    avg_reward : float
        平均累积奖励。
    """
    total_rewards = []
    n_states = env.n_states
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = state_to_onehot(state, n_states).unsqueeze(0)
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()
            
            state, reward, done = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)

def render_policy(env: CliffWalkingEnv, q_network: QNetwork) -> None:
    """在终端渲染学到的策略轨迹。"""
    action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    state = env.reset()
    env.render()
    done = False
    steps = 0
    n_states = env.n_states
    
    while not done and steps < 100:
        with torch.no_grad():
            state_tensor = state_to_onehot(state, n_states).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = q_values.argmax().item()
        
        print(f"动作: {action_names[action]}")
        state, reward, done = env.step(action)
        env.render()
        steps += 1
        
        if done:
            if state == 47:  # 假设终点是 47
                print("🎉 到达终点！")
            else:
                print("💀 掉下悬崖！")

def main() -> None:
    parser = argparse.ArgumentParser(description="悬崖漫步 — 强化学习训练")
    parser.add_argument("--episodes", type=int, default=500, help="训练轮数")
    parser.add_argument("--eval", action="store_true", help="仅评估（需已有 Q 表）")
    args = parser.parse_args()

    env = CliffWalkingEnv()

    if args.eval:
        # 加载预训练模型（这里简化处理，实际需要保存和加载）
        q_network = train(env, episodes=args.episodes)
        avg = evaluate(env, q_network, episodes=10)
        print(f"\n平均奖励: {avg:.2f}")
        render_policy(env, q_network)
    else:
        q_network = train(env, episodes=args.episodes)
        avg = evaluate(env, q_network, episodes=10)
        print(f"\n训练完成 → 平均奖励: {avg:.2f}")
        render_policy(env, q_network)

if __name__ == "__main__":
    main()