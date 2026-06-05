"""带实时渲染的训练 — 每 eval 帧都会打开 Pygame 窗口展示 agent 当前水平"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from env.aim_trainer_env import AimTrainerEnv

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 500
RENDER_FPS = 30

# 颜色
BG_COLOR = (15, 15, 45)
GRID_COLOR = (30, 30, 70)
TARGET_COLOR = (255, 60, 60)
CROSS_COLOR = (83, 168, 182)
TEXT_COLOR = (200, 200, 200)


def build_reward_param_lines(info):
    """构建本步 reward 分解面板。"""
    return [
        f"Hit Reward: {info['reward_hit']:.3f}",
        f"Progress Reward: {info['reward_progress']:.3f}",
        f"Delta Reward: {info['reward_distance_delta']:.3f}",
        f"Time Penalty: {info['reward_time_penalty']:.3f}",
        f"Idle Penalty: {info['reward_idle_penalty']:.3f}",
    ]


def render_frame(
    screen, env, step_count, total_hits, max_steps, total_reward, reward_info, font
):
    """在 Pygame 窗口中渲染 AimTrainerEnv 的当前状态"""
    screen.fill(BG_COLOR)

    # 网格
    for i in range(0, WINDOW_SIZE, 40):
        pygame.draw.line(screen, GRID_COLOR, (i, 0), (i, WINDOW_SIZE))
        pygame.draw.line(screen, GRID_COLOR, (0, i), (WINDOW_SIZE, i))

    # 目标
    for i in range(env.n_targets):
        if not env.target_alive[i]:
            continue
        tx, ty = env.targets[i]
        px = int(tx * WINDOW_SIZE)
        py = int(ty * WINDOW_SIZE)
        r = int(env.target_radius * WINDOW_SIZE)

        for glow in range(r + 8, r - 2, -2):
            c = (255, max(0, 120 - glow * 8), max(0, 120 - glow * 8))
            pygame.draw.circle(screen, c, (px, py), max(1, glow))
        pygame.draw.circle(screen, TARGET_COLOR, (px, py), r, 2)
        pygame.draw.circle(screen, TARGET_COLOR, (px, py), max(1, r // 2), 1)

    # 准星
    cx = int(env.crosshair[0] * WINDOW_SIZE)
    cy = int(env.crosshair[1] * WINDOW_SIZE)
    sz = 12
    pygame.draw.line(screen, CROSS_COLOR, (cx - sz, cy), (cx + sz, cy), 2)
    pygame.draw.line(screen, CROSS_COLOR, (cx, cy - sz), (cx, cy + sz), 2)
    pygame.draw.circle(
        screen, CROSS_COLOR, (cx, cy), int(env.target_radius * WINDOW_SIZE), 1
    )

    # HUD
    hud_lines = [
        (f"Step: {step_count}/{max_steps}", TEXT_COLOR),
        (f"Hits: {total_hits}", TEXT_COLOR),
        (f"Reward: {total_reward:.1f}", CROSS_COLOR),
    ]
    for idx, (line, color) in enumerate(hud_lines):
        text = font.render(line, True, TEXT_COLOR)
        if color is CROSS_COLOR:
            text = font.render(line, True, CROSS_COLOR)
        screen.blit(text, (8, 8 + idx * 22))
    for idx, line in enumerate(build_reward_param_lines(reward_info)):
        text = font.render(line, True, TEXT_COLOR)
        screen.blit(text, (8, 80 + idx * 18))

    pygame.display.flip()


def render_episode_with_model(
    model, raw_env, vec_normalize, screen, font, max_steps=500
):
    """用当前模型跑一个 episode 并在 Pygame 窗口中渲染。返回总命中数

    vec_normalize: 训练时用的 VecNormalize 包装，用于归一化观测
    """
    clock = pygame.time.Clock()

    obs, _ = raw_env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while not done and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # 用 VecNormalize 归一化观测后再预测
        norm_obs = vec_normalize.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, term, trunc, info = raw_env.step(action)
        total_reward += reward
        done = term or trunc
        step += 1

        render_frame(
            screen,
            raw_env,
            step,
            info["total_hits"],
            raw_env.max_steps,
            total_reward,
            info,
            font,
        )
        clock.tick(RENDER_FPS)

    raw_env.close()
    return info["total_hits"], total_reward


def make_env(render_mode=None, target_radius=0.06):
    return AimTrainerEnv(
        max_steps=500,
        n_targets=10,
        target_radius=target_radius,
        render_mode=render_mode,
        hit_reward=500.0,
        time_penalty=5,
        progress_coef=0.25,
        idle_threshold=5,
        efficiency_coef=3.0,
        time_ramp=4.0,
        distance_delta_coef=1.0,
    )


# ── 回调：定期暂停训练、渲染评估 agent ──
class LiveRenderCallback(BaseCallback):
    def __init__(
        self,
        vec_normalize,
        target_radius,
        render_freq=10_000,
        max_render_steps=None,
        verbose=0,
    ):
        super().__init__(verbose)
        self.vec_normalize = vec_normalize  # 用于归一化观测
        self.target_radius = target_radius
        self.render_freq = render_freq
        self.max_render_steps = max_render_steps
        self.screen = None
        self.font = None

    def _show_eval(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Aim Trainer — RL Agent Demo")
            self.font = pygame.font.SysFont("segoeui,Arial,PingFangSC", 16)

        # 用裸环境渲染，但将 VecNormalize 传给推理使用
        raw_env = make_env(target_radius=self.target_radius)
        render_steps = (
            self.max_render_steps
            if self.max_render_steps is not None
            else raw_env.max_steps
        )
        hits, total_reward = render_episode_with_model(
            self.model,
            raw_env,
            self.vec_normalize,
            self.screen,
            self.font,
            max_steps=render_steps,
        )
        print(
            f"  🎮 [Demo @ {self.num_timesteps:,} steps] Agent 命中 {hits} 次, 累计 reward: {total_reward:.1f}"
        )

    def _on_step(self) -> bool:
        if self.render_freq > 0 and self.num_timesteps % self.render_freq == 0:
            self._show_eval()
        return True

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None


def train(total_timesteps=200_000, render_freq=10_000, target_radius=0.06):
    print(f"🎮 实时可视化训练 | {total_timesteps:,} 步")
    print(f"   每 {render_freq:,} 步暂停并播放 agent 当前水平\n")
    print(f"   target_radius={target_radius}\n")

    train_env = DummyVecEnv([lambda: make_env(target_radius=target_radius)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.01,
        device="auto",
    )

    render_cb = LiveRenderCallback(
        train_env,
        target_radius=target_radius,
        render_freq=render_freq,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=render_cb)
    except KeyboardInterrupt:
        print("\n⏸️  训练中断")
    except SystemExit:
        pass
    finally:
        render_cb.close()

    model_path = ARTIFACTS_DIR / "ppo_aim_trainer.zip"
    model.save(str(model_path))
    train_env.save(str(ARTIFACTS_DIR / "vec_normalize_stats.pkl"))
    print(f"\n✅ 模型已保存: {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="可视化训练 Aim Trainer")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument(
        "--render-freq",
        type=int,
        default=10_000,
        help="每隔多少步渲染一次 agent 演示（0 表示不渲染）",
    )
    parser.add_argument(
        "--target-radius", type=float, default=0.06, help="目标命中半径"
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.steps,
        render_freq=args.render_freq,
        target_radius=args.target_radius,
    )
