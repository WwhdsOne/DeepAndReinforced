"""悬崖漫步训练入口 —— 在此实现你的强化学习算法。

提供骨架结构：环境创建、训练循环、评估、命令行参数。
"""

import argparse
import numpy as np

from env import CliffWalkingEnv


def train(env: CliffWalkingEnv, episodes: int = 500) -> np.ndarray:
    """训练主函数 —— 在此实现你的算法。

    Parameters
    ----------
    env : CliffWalkingEnv
        悬崖漫步环境。
    episodes : int
        训练轮数。

    Returns
    -------
    Q : np.ndarray
        训练后的 Q 表，形状 (48, 4)。
    """
    n_states = env.n_states
    n_actions = env.n_actions

    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        state = env.reset()
        done = False

        if np.random.random() < 0.1:
            action = np.random.randint(0, n_actions)
        else:
            action = Q[state].argmax()

        while not done:
            next_state, reward, done = env.step(action)

            if not done:
                if np.random.random() < 0.1:
                    next_action = np.random.randint(0, n_actions)
                else:
                    next_action = Q[next_state].argmax()
                td_target = reward + 0.9 * Q[next_state, next_action]
            else:
                td_target = reward

            Q[state, action] = Q[state, action] + 0.1 * (td_target - Q[state, action])

            state = next_state
            action = next_action

    return Q


def evaluate(env: CliffWalkingEnv, Q: np.ndarray, episodes: int = 10) -> float:
    """评估训练后的策略。

    Parameters
    ----------
    env : CliffWalkingEnv
    Q : np.ndarray
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


def render_policy(env: CliffWalkingEnv, Q: np.ndarray) -> None:
    """在终端渲染学到的策略轨迹。"""
    action_names = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    state = env.reset()
    env.render()
    done = False
    steps = 0
    while not done and steps < 100:
        action = Q[state].argmax()
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
    print(Q)


if __name__ == "__main__":
    main()
