import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO


class AimTrainerEnv(gym.Env):
    """多目标瞄准仿真环境

    状态：[crosshair_x, crosshair_y, t1_x, t1_y, t1_alive, ...]
    动作：[dx, dy]  连续位移，进入目标半径自动命中

    参数
    ----
    max_steps : int      每个 episode 的最大步数
    n_targets : int      同时存在的目标数量
    target_radius : float 命中判定的目标半径
    hit_reward : float   命中奖励（默认 10.0）
    time_penalty : float 每步基础时间惩罚（默认 0.01）
    time_ramp : float     终点惩罚/起点惩罚的倍数，越大越紧迫（默认 5.0）
    progress_coef : float 进步奖励系数（默认 0.2）
    idle_threshold : float | None 发呆判定阈值；默认按动作步长自适应
    efficiency_coef : float 路径效率系数，命中越快额外奖励越高（默认 2.0）
    action_step : float | None  最大位移步长（默认自动=max(target_radius×0.4, 0.004)）
    min_action_step : float 动作步长下限，避免小目标时移动过慢（默认 0.004）
    near_zone_scale : float 靠近奖励半径相对 target_radius 的倍数（默认 3.0）
    near_zone_min : float 靠近奖励半径下限（默认 0.06）
    distance_delta_coef : float 每步距离变化奖励系数，朝目标更近给正反馈，变远给负反馈（默认 0.6）
    regress_penalty_scale : float 远离目标时的额外惩罚倍数（默认 1.5）
    render_mode : str | None  渲染模式 ("human" / "rgb_array")
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, max_steps=1600, n_targets=5, target_radius=0.04,
                 hit_reward=10.0, time_penalty=0.01, time_ramp=5.0,
                 progress_coef=0.2, idle_threshold=None,
                 efficiency_coef=2.0, action_step=None,
                 min_action_step=0.004, near_zone_scale=3.0,
                 near_zone_min=0.06, distance_delta_coef=0.6,
                 regress_penalty_scale=1.5,
                 render_mode=None):
        super().__init__()

        self.max_steps = max_steps
        self.n_targets = n_targets
        self.target_radius = target_radius
        self.hit_reward = hit_reward
        self.time_penalty = time_penalty
        self.time_ramp = time_ramp
        self.progress_coef = progress_coef
        self.efficiency_coef = efficiency_coef
        self.min_action_step = min_action_step
        self.near_zone_scale = near_zone_scale
        self.near_zone_min = near_zone_min
        self.distance_delta_coef = distance_delta_coef
        self.regress_penalty_scale = regress_penalty_scale

        # 状态：[crosshair_xy(2), target_xy_alive * n_targets(3 each)]
        obs_size = 2 + n_targets * 3 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # 动作：连续位移。给小靶子保留最小步长，避免移动过细导致到达太慢。
        default_action_step = max(target_radius * 0.4, min_action_step)
        self.action_step = action_step if action_step is not None else default_action_step
        # 发呆阈值默认按动作尺度自适应，避免小靶子微调动作被误判为发呆。
        self.idle_threshold = (
            idle_threshold if idle_threshold is not None
            else min(0.002, self.action_step * 0.15)
        )
        self.action_space = spaces.Box(
            low=-self.action_step, high=self.action_step, shape=(2,), dtype=np.float32
        )

        self.render_mode = render_mode

        # 内部状态
        self.crosshair = np.zeros(2, dtype=np.float32)
        self.targets = np.zeros((n_targets, 2), dtype=np.float32)
        self.target_alive = np.zeros(n_targets, dtype=bool)
        self.current_step = 0
        self.total_hits = 0
        self._steps_since_last_hit = 0
        self._proximity_claimed = np.zeros(n_targets, dtype=bool)  # 防止重复蹭分
        self._best_nearest_distance = np.inf
        self._prev_nearest_distance = np.inf

        # 渲染缓存
        self._fig = None
        self._ax = None

    # ── 内部辅助 ──────────────────────────────────────────

    def _spawn_target(self, idx):
        """在 idx 位置随机生成一个目标。"""
        # 目标位置范围 [0.15, 0.85] 避免贴边
        self.targets[idx] = self.np_random.uniform(0.15, 0.85, size=2).astype(
            np.float32
        )
        self.target_alive[idx] = True

    def _get_obs(self):
        """构建观测向量。"""
        obs = np.zeros(2 + self.n_targets * 3 + 2, dtype=np.float32)
        obs[0:2] = self.crosshair
        for i in range(self.n_targets):
            base = 2 + i * 3
            if self.target_alive[i]:
                obs[base : base + 2] = self.targets[i]
                obs[base + 2] = 1.0
        alive_idx = np.where(self.target_alive)[0]
        if len(alive_idx) > 0:
            diffs = self.targets[alive_idx] - self.crosshair
            nearest_idx_in_alive = int(np.argmin(np.sqrt(np.sum(diffs ** 2, axis=1))))
            obs[-2:] = diffs[nearest_idx_in_alive]
        return obs

    # ── Gymnasium 接口 ────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.total_hits = 0
        self._steps_since_last_hit = 0
        self._proximity_claimed[:] = False
        self._best_nearest_distance = np.inf
        self._prev_nearest_distance = np.inf
        self.crosshair = np.array([0.5, 0.5], dtype=np.float32)

        for i in range(self.n_targets):
            self._spawn_target(i)

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 移动准星并 clamp
        action_arr = np.asarray(action, dtype=np.float32)
        old_crosshair = self.crosshair.copy()
        alive_idx = np.where(self.target_alive)[0]
        self.crosshair = np.clip(self.crosshair + action_arr, 0.0, 1.0)
        actual_delta = self.crosshair - old_crosshair

        # 发呆惩罚：按实际位移计算，顶墙或卡边时也算无效动作。
        action_norm = float(np.sqrt(np.sum(actual_delta ** 2)))
        idle_penalty = 0.0
        if action_norm < self.idle_threshold:
            idle_penalty = -self.time_penalty * 5  # 发呆比普通走步更贵

        reward = 0.0
        progress_reward = 0.0
        distance_delta_reward = 0.0
        hit_reward_total = 0.0
        hit_count = 0
        self._steps_since_last_hit += 1

        # 靠近奖励：扩大引导圈，并奖励朝最近目标的持续逼近。
        if len(alive_idx) > 0:
            diffs_all = self.crosshair - self.targets[alive_idx]
            dists = np.sqrt(np.sum(diffs_all ** 2, axis=1))
            nearest_idx_in_alive = int(np.argmin(dists))
            nearest_idx = alive_idx[nearest_idx_in_alive]
            nearest = float(dists[nearest_idx_in_alive])
            near_zone = max(
                self.target_radius * self.near_zone_scale,
                self.near_zone_min,
            )
            if np.isfinite(self._prev_nearest_distance):
                distance_delta = (self._prev_nearest_distance - nearest) / near_zone
                distance_delta_reward = self.distance_delta_coef * distance_delta
                if distance_delta_reward < 0.0:
                    distance_delta_reward *= self.regress_penalty_scale
                reward += distance_delta_reward
            if np.isfinite(self._best_nearest_distance) and nearest < self._best_nearest_distance:
                improvement = self._best_nearest_distance - nearest
                progress_reward += self.progress_coef * (improvement / near_zone)
                reward += self.progress_coef * (improvement / near_zone)
            self._best_nearest_distance = nearest
            self._prev_nearest_distance = nearest
            if nearest < near_zone and not self._proximity_claimed[nearest_idx]:
                progress_reward += self.progress_coef * (1.0 - nearest / near_zone)
                reward += self.progress_coef * (1.0 - nearest / near_zone)
                self._proximity_claimed[nearest_idx] = True

        # 命中检测
        for i in range(self.n_targets):
            if not self.target_alive[i]:
                continue
            dist = float(np.sqrt(np.sum((self.crosshair - self.targets[i]) ** 2)))
            if dist < self.target_radius:
                # 路径效率奖励：步数越少额外分越高
                hit_bonus = self.hit_reward + self.efficiency_coef / max(1, self._steps_since_last_hit)
                hit_reward_total += hit_bonus
                reward += hit_bonus
                hit_count += 1
                self.total_hits += 1
                self.target_alive[i] = False
                self._spawn_target(i)
                self._steps_since_last_hit = 0  # 重置计时
                self._proximity_claimed[i] = False  # 新目标可再次拿靠近奖励
                self._best_nearest_distance = np.inf
                self._prev_nearest_distance = np.inf

        # 时间惩罚（阶梯递增：越晚越贵，催促快速命中）
        progress_ratio = self.current_step / max(self.max_steps, 1)
        ramp = 1.0 + (self.time_ramp - 1.0) * progress_ratio
        time_penalty_reward = -self.time_penalty * ramp
        reward += time_penalty_reward

        # 发呆惩罚
        reward += idle_penalty

        terminated = False
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "hit_count": hit_count,
            "total_hits": self.total_hits,
            "reward_total": reward,
            "reward_progress": progress_reward,
            "reward_distance_delta": distance_delta_reward,
            "reward_hit": hit_reward_total,
            "reward_time_penalty": time_penalty_reward,
            "reward_idle_penalty": idle_penalty,
        }

    def render(self):
        if self.render_mode is None:
            return None

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        self._ax.clear()
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._ax.set_aspect("equal")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_title(
            f"Aim Trainer  |  Step {self.current_step}/{self.max_steps}"
        )

        # 目标
        for i in range(self.n_targets):
            if self.target_alive[i]:
                c = patches.Circle(
                    self.targets[i],
                    self.target_radius,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.35,
                )
                self._ax.add_patch(c)

        # 准星（十字线）
        ch_x, ch_y = self.crosshair
        self._ax.plot(
            ch_x, ch_y, "b+", markersize=14, markeredgewidth=2, label="crosshair"
        )
        self._ax.axhline(
            y=ch_y, xmin=0, xmax=1, color="blue", alpha=0.25, linewidth=0.8
        )
        self._ax.axvline(
            x=ch_x, ymin=0, ymax=1, color="blue", alpha=0.25, linewidth=0.8
        )

        if self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            buf = BytesIO()
            self._fig.savefig(buf, format="png", dpi=50, bbox_inches="tight")
            buf.seek(0)
            img = plt.imread(buf)
            buf.close()
            return (img * 255).astype(np.uint8)
        elif self.render_mode == "human":
            plt.pause(0.02)

        return None

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
