"""Aim Trainer 评估脚本 — 加载模型并可视化推理"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.aim_trainer_env import AimTrainerEnv

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "ppo_aim_trainer.zip"
STATS_PATH = ARTIFACTS_DIR / "vec_normalize_stats.pkl"


def make_env(render_mode=None):
    return AimTrainerEnv(
        max_steps=500, n_targets=5, target_radius=0.04, render_mode=render_mode
    )


def evaluate(n_episodes=10, render_gif=False):
    if not MODEL_PATH.exists():
        print(f"模型文件不存在: {MODEL_PATH}")
        print("请先运行 src/train.py 训练模型")
        return

    # 加载模型
    model = PPO.load(str(MODEL_PATH))

    # 重建环境包装
    env = DummyVecEnv([make_env])
    if STATS_PATH.exists():
        env = VecNormalize.load(str(STATS_PATH), env)
        env.training = False
        env.norm_reward = False

    print(f"评估 {n_episodes} 个 episode...\n")

    total_hits = 0
    episode_hits = []
    episode_steps = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_hits = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = env.step(action)
            done = done_arr[0]
            # 累计命中数
            if info_arr and "total_hits" in info_arr[0]:
                ep_hits = info_arr[0]["total_hits"]
            steps += 1

        episode_hits.append(ep_hits)
        episode_steps.append(steps)
        total_hits += ep_hits
        print(f"  Episode {ep+1:2d}: {ep_hits:4d} hits | {steps:4d} steps")

    avg_hits = np.mean(episode_hits)
    avg_steps = np.mean(episode_steps)

    # 统计命中时每步平均命中率
    hits_per_step = total_hits / (n_episodes * 500)
    print(f"\n{'='*40}")
    print(f"平均命中数:   {avg_hits:.1f}")
    print(f"平均步数:     {avg_steps:.0f}")
    print(f"命中率:       {hits_per_step:.3f} hits/step")
    print(f"总计命中:     {total_hits}")
    print(f"{'='*40}")

    # 可选：录制最后 1 个 episode 为 GIF
    if render_gif:
        _render_gif(model, env)


def _render_gif(model, env):
    """录制一个 episode 并保存为 GIF。"""
    try:
        import imageio
    except ImportError:
        print("需要 imageio 来生成 GIF。安装: `uv pip install imageio`")
        return

    gif_path = ARTIFACTS_DIR / "eval_episode.gif"
    render_env = make_env(render_mode="rgb_array")

    obs, _ = render_env.reset()
    frames = [render_env.render()]
    done = False

    while not done:
        # 重建 VecNormalize 观测
        obs_vec = env.reset()
        action, _ = model.predict(obs_vec, deterministic=True)
        obs, reward, term, trunc, _ = render_env.step(action[0])
        done = term or trunc
        frames.append(render_env.render())

    render_env.close()
    imageio.mimsave(str(gif_path), frames, fps=15)
    print(f"GIF 已保存到 {gif_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="评估 episode 数")
    parser.add_argument("--gif", action="store_true", help="录制 GIF")
    args = parser.parse_args()

    evaluate(n_episodes=args.episodes, render_gif=args.gif)
