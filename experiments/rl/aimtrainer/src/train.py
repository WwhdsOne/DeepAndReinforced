"""Aim Trainer 训练脚本 — 使用 Stable-Baselines3 PPO"""

import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from env.aim_trainer_env import AimTrainerEnv

# ── 路径配置 ──────────────────────────────────────────────

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def make_env(render_mode=None, target_radius=0.04):
    """工厂函数：创建一个 AimTrainerEnv 实例。"""
    return AimTrainerEnv(
        max_steps=1600,
        n_targets=5,
        target_radius=target_radius,
        render_mode=render_mode,
    )


def train(total_timesteps=200_000, eval_freq=10_000, target_radius=0.04):
    print(
        f"训练配置：total_timesteps={total_timesteps}, eval_freq={eval_freq}, target_radius={target_radius}"
    )
    print(f"日志目录：{LOG_DIR}")

    # ── 训练环境 + VecNormalize ──────────────────────────
    train_env = DummyVecEnv([lambda: make_env(target_radius=target_radius)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # ── 评估环境 ─────────────────────────────────────────
    eval_env = DummyVecEnv([lambda: make_env(target_radius=target_radius)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # ── 回调：定期评估 ──────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ARTIFACTS_DIR),
        log_path=str(LOG_DIR),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # ── PPO 模型 ─────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.1,
        device="auto",
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
        )
    except KeyboardInterrupt:
        print("\n训练中断，正在保存模型...")

    # ── 保存模型 ─────────────────────────────────────────
    model_path = ARTIFACTS_DIR / "ppo_aim_trainer.zip"
    model.save(str(model_path))
    # 同时保存 VecNormalize 统计量
    train_env.save(str(ARTIFACTS_DIR / "vec_normalize_stats.pkl"))
    print(f"模型已保存到 {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000, help="总训练步数")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="评估频率")
    parser.add_argument(
        "--target-radius", type=float, default=0.04, help="目标命中半径"
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.steps,
        eval_freq=args.eval_freq,
        target_radius=args.target_radius,
    )
