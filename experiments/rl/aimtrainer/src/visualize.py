"""瞄准仿真可视化——包含训练曲线绘制与推理过程录制

1. 训练完成后绘制 reward / loss 曲线图
2. 加载模型并以人眼可见方式展示推理过程
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# ── 路径配置 ────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "ppo_aim_trainer.zip"
STATS_PATH = ARTIFACTS_DIR / "vec_normalize_stats.pkl"


# ========================== 回调：记录训练指标 ==========================


class RecordTrainingStats(BaseCallback):
    """记录每一步的损失与每个评估点上的平均得分，用于后续画图。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > 0:
            rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            self.episode_rewards.append(rew)
            self.timesteps.append(self.num_timesteps)
        return True


# ========================== 训练曲线图 ============================


def plot_curves(log_dir=None):
    """读取 SB3 日志或者已保存的训练统计，绘制 reward / episode length 曲线。

    参数
    ----
    log_dir : Path | None
        monitor.csv 文件所在目录（EvalCallback 自动生成）
    """
    import pandas as pd

    csv_files = list(ARTIFACTS_DIR.glob("logs/*/progress.csv"))
    if not csv_files:
        print("没有找到 SB3 评估日志（logs/*/progress.csv）。")
        print("先运行训练脚本以获得评估数据：")
        print("  python src/train.py --steps 50000")
        return

    df = pd.read_csv(csv_files[-1])  # 取最新的

    # 筛选 eval/ 开头的列
    eval_cols = [c for c in df.columns if c.startswith("eval/")]
    if not eval_cols:
        print("日志中没有 eval/ 开头的列，请确保训练脚本中 eval_freq > 0")
        return

    # 取 'time/total_timesteps' 作为 x 轴
    x_axis = (
        df["time/total_timesteps"]
        if "time/total_timesteps" in df.columns
        else range(len(df))
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 子图 1：平均奖励
    if "eval/mean_reward" in df.columns:
        axes[0].plot(x_axis, df["eval/mean_reward"], "b-", linewidth=2)
        axes[0].fill_between(
            x_axis,
            df["eval/mean_reward"] - df.get("eval/std_reward", 0),
            df["eval/mean_reward"] + df.get("eval/std_reward", 0),
            alpha=0.2,
            color="blue",
        )
        axes[0].set_title("Eval Mean Reward")
        axes[0].set_xlabel("Timesteps")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No eval data", ha="center", va="center")

    # 子图 2：平均 episode 长度
    if "eval/mean_ep_length" in df.columns:
        axes[1].plot(x_axis, df["eval/mean_ep_length"], "orange", linewidth=2)
        axes[1].set_title("Eval Episode Length")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No eval data", ha="center", va="center")

    fig.suptitle("Aim Trainer — PPO Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = ARTIFACTS_DIR / "training_curves.png"
    fig.savefig(save_path, dpi=120)
    print(f"训练曲线已保存到 {save_path}")
    plt.show()


# ========================== 实时推理回放 ============================


def play_episode(delay=0.02):
    """使用训练好的模型运行一个 episode，并实时渲染。

    参数
    ----
    delay : float
        每帧之间的暂停秒数（越小越快），用作模拟速度倍率
    """
    if not MODEL_PATH.exists():
        print(f"模型文件不存在: {MODEL_PATH}")
        print("请先运行训练脚本：")
        print("  python src/train.py --steps 100000")
        return

    # 加载模型
    model = PPO.load(str(MODEL_PATH))

    # 重建环境包装（推理用的）
    env = DummyVecEnv([lambda: gym.make("AimTrainerEnv-v0")])

    # 创建带渲染的环境
    render_env = gym.make("AimTrainerEnv-v0", render_mode="human")
    # 手动调整渲染频率
    AimTrainerEnv_ = type(render_env)

    # 先用裸环境 reset
    obs_render, _ = render_env.reset()

    # VecNormalize 只需加载统计量作用在观测上
    if STATS_PATH.exists():
        env = VecNormalize.load(str(STATS_PATH), env)
        env.training = False
        env.norm_reward = False

    # 注意：AimTrainerEnv-v0 如果没注册，我们会手动创建
    print("回放开始 … 按 Ctrl+C 可中断\n")

    try:
        done = False
        step = 0
        while not done:
            # 将 render_env 的观测传给 VecNormalize 做标准化
            obs_vec = (
                env.get_original_obs()
                if hasattr(env, "get_original_obs")
                else env.reset()
            )
            # 简单做法：用 VecNormalize._normalize_observation()
            # 更简单做法：用原始观测直接用模型预测，因为我们保存的是原始环境的模型
            action, _ = model.predict(obs_render, deterministic=True)
            obs_render, reward, term, trunc, _info = render_env.step(action)
            done = term or trunc
            step += 1

            # 实时刷新
            render_env.render()
            plt.pause(delay)

    except KeyboardInterrupt:
        print("\n回放已由用户中断。")
    finally:
        render_env.close()
        print(f"\nEpisode 结束，共 {step} 步")


# ========================== 自由探索（随机代理） ============================


def random_agent():
    """随机代理作为基线对比——随机移动准星。"""
    print("随机代理演示（基线对比）…\n按 Ctrl+C 可中断\n")
    env = gym.make("AimTrainerEnv-v0", render_mode="human")
    env.reset()
    total_hits = 0

    try:
        done = False
        while not done:
            action = env.action_space.sample()
            _, reward, term, trunc, info = env.step(action)
            total_hits += info["hit_count"]  # 累加这一帧的命中数
            done = term or trunc

            env.render()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n已中断。")
    finally:
        env.close()

    print(f"随机代理总命中数: {total_hits}")


# ========================== 入口 ============================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aim Trainer 可视化")
    sub = parser.add_subparsers(dest="command")

    # train + plot 曲线
    sub_train = sub.add_parser("train", help="从头训练并画曲线")
    sub_train.add_argument("--steps", type=int, default=100000)

    # plot 单独画曲线（已有训练数据时）
    sub_plot = sub.add_parser("plot", help="仅从已有日志绘制训练曲线")

    # 推理回放
    sub_play = sub.add_parser("play", help="加载模型进行推理回放")
    sub_play.add_argument(
        "--speed", type=float, default=0.02, help="帧间隔秒数 (越小越快)"
    )

    # 随机代理
    sub_rand = sub.add_parser("random", help="随机代理演示")

    # 训练 + eval 一站式
    sub_all = sub.add_parser("all", help="训练→画曲线→回放")
    sub_all.add_argument("--steps", type=int, default=100000)

    args = parser.parse_args()

    # ── 为方便使用，若 AimTrainerEnv 未注册到 gym 则手动注册 ──
    from env.aim_trainer_env import AimTrainerEnv

    if "AimTrainerEnv-v0" not in gym.envs.registry:
        gym.envs.registration.register(
            id="AimTrainerEnv-v0",
            entry_point="env.aim_trainer_env:AimTrainerEnv",
        )

    if args.command == "train":
        plot_curves()
    elif args.command == "plot":
        plot_curves()
    elif args.command == "play":
        play_episode(delay=args.speed)
    elif args.command == "random":
        random_agent()
    elif args.command == "all":
        print("=" * 50)
        print("1）开始训练…")
        print("=" * 50)
        from src.train import train

        train(total_timesteps=args.steps, eval_freq=args.steps // 5)
        print("\n" + "=" * 50)
        print("2）绘制训练曲线…")
        print("=" * 50)
        plot_curves()
        print("\n" + "=" * 50)
        print("3）推理回放…")
        print("=" * 50)
        play_episode(delay=0.02)
    else:
        parser.print_help()
