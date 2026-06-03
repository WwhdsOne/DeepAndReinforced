"""AI 演示服务器 — 用 eel 连接训练好的模型和浏览器游戏。

启动方式：
    uv run python experiments/rl/aimtrainer/src/ai_play.py

原理：
    - Python 端加载 PPO 模型 + VecNormalize，运行 AimTrainerEnv
    - 每步推理后将十字准星、目标、奖励等状态通过 eel 推送到浏览器
    - 浏览器纯做渲染，不参与游戏逻辑
"""

from __future__ import annotations

import sys
from pathlib import Path

import eel
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 确保项目根目录和 aimtrainer 目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_AIMTRAINER_DIR = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT, str(_AIMTRAINER_DIR)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from env.aim_trainer_env import AimTrainerEnv  # noqa: E402

# ── 路径 ────────────────────────────────────────────────
ARTIFACTS_DIR = _AIMTRAINER_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "ppo_aim_trainer.zip"
STATS_PATH = ARTIFACTS_DIR / "vec_normalize_stats.pkl"
HTML_DIR = _AIMTRAINER_DIR / "src"

# ── 全局状态 ────────────────────────────────────────────
_model: PPO | None = None
_env: AimTrainerEnv | None = None
_episode_running: bool = False
_stop_requested: bool = False
_vec_normalize = None  # VecNormalize 包装器（用于观测归一化）


# ══════════════════════════════════════════════════════════
#  eel 暴露给 JS 的函数
# ══════════════════════════════════════════════════════════

@eel.expose
def ai_start_episode(target_radius: float = 0.04, max_steps: int = 500) -> None:
    """JS 调用：开始一个 AI 推理 episode。"""
    global _env, _vec_normalize, _episode_running, _stop_requested

    if _env is not None:
        _env.close()

    # 根据模型的 obs 空间自动推断 n_targets
    # obs_size = 2 + n_targets * 3 + 2，所以 n_targets = (obs_size - 4) / 3
    if _model is not None:
        obs_size = _model.observation_space.shape[0]
        n_targets = int((obs_size - 4) // 3)
    else:
        n_targets = 5

    raw_env = AimTrainerEnv(
        max_steps=max_steps,
        n_targets=n_targets,
        target_radius=target_radius,
        render_mode=None,
    )

    # 用 VecNormalize 包装（与训练时一致），加载统计量
    _env = raw_env  # 保留原始环境引用，用于 _push_state 读取内部状态
    venv = DummyVecEnv([lambda: raw_env])
    if STATS_PATH.exists():
        _vec_normalize = VecNormalize.load(str(STATS_PATH), venv)
        _vec_normalize.training = False
        _vec_normalize.norm_reward = False
    else:
        _vec_normalize = venv

    obs = _vec_normalize.reset()
    _episode_running = True
    _stop_requested = False

    # 推送初始状态
    _push_state(hits=0, total_reward=0.0)

    # 运行游戏循环（每步 sleep 交出控制权给 eel）
    while not _stop_requested:
        # 模型推理（obs 已经过 VecNormalize 归一化）
        if _model is not None:
            action, _ = _model.predict(obs, deterministic=True)
        else:
            action = np.zeros(2, dtype=np.float32)

        obs, reward, done_arr, info_arr = _vec_normalize.step(action)
        done = done_arr[0]

        _push_state(
            hits=info_arr[0].get("total_hits", 0) if info_arr else 0,
            total_reward=info_arr[0].get("reward_total", 0.0) if info_arr else 0.0,
        )

        if done:
            break

        eel.sleep(0.03)  # ~30 FPS

    _episode_running = False
    eel.ai_episode_done()


@eel.expose
def ai_stop_episode() -> None:
    """JS 调用：停止当前 episode。"""
    global _stop_requested
    _stop_requested = True


@eel.expose
def ai_get_artifacts_info() -> dict:
    """JS 调用：获取模型/环境参数信息。"""
    if _model is not None:
        obs_size = _model.observation_space.shape[0]
        n_targets = int((obs_size - 4) // 3)
    else:
        n_targets = 5

    return {
        "model_exists": MODEL_PATH.exists(),
        "stats_exists": STATS_PATH.exists(),
        "model_name": MODEL_PATH.name,
        "target_radius": 0.04,
        "max_steps": 500,
        "n_targets": n_targets,
    }


# ══════════════════════════════════════════════════════════
#  内部辅助
# ══════════════════════════════════════════════════════════

def _push_state(hits: int, total_reward: float) -> None:
    """将当前环境状态推送到浏览器渲染。"""
    if _env is None:
        return

    targets_json = []
    for i in range(_env.n_targets):
        targets_json.append({
            "x": float(_env.targets[i][0]),
            "y": float(_env.targets[i][1]),
            "alive": bool(_env.target_alive[i]),
        })

    # 锁定目标信息
    locked_idx = int(_env._locked_target_idx)
    near_zone = max(
        _env.target_radius * _env.near_zone_scale,
        _env.near_zone_min,
    )
    locked_dist = -1.0
    if locked_idx >= 0 and _env.target_alive[locked_idx]:
        locked_dist = float(np.sqrt(np.sum(
            (_env.crosshair - _env.targets[locked_idx]) ** 2
        )))

    eel.render_state({
        "crosshair_x": float(_env.crosshair[0]),
        "crosshair_y": float(_env.crosshair[1]),
        "targets": targets_json,
        "step": int(_env.current_step),
        "max_steps": int(_env.max_steps),
        "hits": hits,
        "total_reward": round(total_reward, 3),
        "locked_idx": locked_idx,
        "locked_dist": round(locked_dist, 4),
        "target_radius": float(_env.target_radius),
        "near_zone": float(near_zone),
    })


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

def main():
    global _model

    print("=" * 50)
    print("AI 瞄准训练器 — 浏览器演示")
    print("=" * 50)

    # 加载模型
    if MODEL_PATH.exists():
        _model = PPO.load(str(MODEL_PATH))
        print(f"模型已加载: {MODEL_PATH}")
    else:
        print(f"⚠️  模型不存在: {MODEL_PATH}，将使用随机动作")
        print("   请先运行 train.py 训练模型")

    # 初始化 eel
    eel.init(str(HTML_DIR))

    # 启动（在默认浏览器中打开）
    print(f"\n正在启动浏览器...")
    eel.start(
        "aim_trainer_game.html",
        mode="chrome",  # 尝试用 chrome 打开，失败则用默认浏览器
        size=(750, 780),
        port=0,  # 随机端口
        block=True,
    )


if __name__ == "__main__":
    main()
