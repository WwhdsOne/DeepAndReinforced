import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from env.aim_trainer_env import AimTrainerEnv


class TestAimTrainerEnv:
    """AimTrainerEnv 单元测试"""

    @pytest.fixture
    def env(self):
        return AimTrainerEnv(max_steps=100, n_targets=3)

    def test_reset(self, env):
        obs, info = env.reset()
        assert obs.shape == (2 + 3 * 3 + 2,)
        assert np.all(obs[2:5] > 0)  # 第一个目标存在
        assert obs[4] == 1.0  # 第一个目标 alive 标志
        assert np.allclose(obs[0:2], [0.5, 0.5])  # 准星初始位置
        assert env.current_step == 0

    def test_step_movement(self, env):
        env.reset()
        action = np.array([0.04, 0.04], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        assert np.allclose(obs[0:2], [0.54, 0.54])
        assert env.current_step == 1

    def test_step_clamp(self, env):
        env.reset()
        env.crosshair = np.array([0.02, 0.98], dtype=np.float32)
        action = np.array([-0.05, 0.05], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        assert np.allclose(obs[0], 0.0)   # clamp left
        assert np.allclose(obs[1], 1.0)   # clamp top

    def test_hit_detection(self, env):
        env.reset()
        # 关掉其他目标，只留一个
        for i in range(env.n_targets):
            env.target_alive[i] = False
        env.crosshair = np.array([0.5, 0.5], dtype=np.float32)
        env.targets[0] = np.array([0.51, 0.51], dtype=np.float32)
        env.target_alive[0] = True
        # 距离 ≈ 0.014，小于 target_radius=0.04，应该命中
        _, reward, _, _, info = env.step(np.array([0.0, 0.0]))
        assert info["hit_count"] == 1
        assert reward > 9.0  # 10 - 0.01 + 距离奖励

    def test_target_respawn_after_hit(self, env):
        env.reset()
        old_target = env.targets[0].copy()
        env.crosshair = old_target.copy()  # 直接放到目标上
        env.step(np.array([0.0, 0.0]))
        assert not np.allclose(env.targets[0], old_target)  # 新目标位置不同

    def test_truncated(self, env):
        env.reset()
        for _ in range(99):
            env.step(env.action_space.sample())
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        assert not term
        assert trunc

    def test_observation_space(self, env):
        obs, _ = env.reset()
        assert obs in env.observation_space
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs in env.observation_space

    def test_observation_contains_nearest_target_relative_vector(self, env):
        env.reset()
        env.crosshair = np.array([0.5, 0.5], dtype=np.float32)
        env.targets[0] = np.array([0.7, 0.4], dtype=np.float32)
        env.targets[1] = np.array([0.55, 0.52], dtype=np.float32)
        env.targets[2] = np.array([0.2, 0.2], dtype=np.float32)
        env.target_alive[:] = True

        obs = env._get_obs()

        assert np.allclose(obs[-2:], [0.05, 0.02], atol=1e-6)

    def test_seed_reproducibility(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = AimTrainerEnv(max_steps=100, n_targets=3).reset(seed=42)
        assert np.allclose(obs1, obs2)

    def test_default_action_step_scales_with_step_budget(self):
        env = AimTrainerEnv(target_radius=0.04)
        assert env.max_steps == 1600
        assert np.isclose(env.action_step, 0.016)

    def test_distance_delta_reward_penalizes_moving_away(self):
        env = AimTrainerEnv(max_steps=100, n_targets=1, target_radius=0.04, lock_on_speed=0.0)
        env.reset(seed=0)
        env.targets[0] = np.array([0.6, 0.5], dtype=np.float32)
        env.target_alive[0] = True
        env.crosshair = np.array([0.5, 0.5], dtype=np.float32)

        _, reward_closer, _, _, _ = env.step(np.array([0.01, 0.0], dtype=np.float32))
        env.crosshair = np.array([0.51, 0.5], dtype=np.float32)
        env._prev_nearest_distance = 0.09
        _, reward_away, _, _, _ = env.step(np.array([-0.01, 0.0], dtype=np.float32))

        assert reward_closer > reward_away

    def test_moving_away_is_penalized_more_than_moving_closer_is_rewarded(self):
        env = AimTrainerEnv(
            max_steps=100,
            n_targets=1,
            target_radius=0.04,
            time_penalty=0.0,
            progress_coef=0.0,
            distance_delta_coef=1.0,
            lock_on_speed=0.0,
        )
        env.reset(seed=0)
        env.targets[0] = np.array([0.6, 0.5], dtype=np.float32)
        env.target_alive[0] = True
        env.crosshair = np.array([0.5, 0.5], dtype=np.float32)
        env._locked_target_idx = 0
        env._prev_locked_distance = 0.1
        env._best_locked_distance = 0.1

        _, reward_closer, _, _, info_closer = env.step(np.array([0.01, 0.0], dtype=np.float32))

        env.crosshair = np.array([0.5, 0.5], dtype=np.float32)
        env._locked_target_idx = 0
        env._prev_locked_distance = 0.1
        env._best_locked_distance = 0.1
        _, reward_away, _, _, info_away = env.step(np.array([-0.01, 0.0], dtype=np.float32))

        assert info_closer["reward_distance_delta"] > 0
        assert info_away["reward_distance_delta"] < 0
        assert abs(info_away["reward_distance_delta"]) >= abs(info_closer["reward_distance_delta"]) * 1.2
        assert abs(reward_away) >= abs(reward_closer) * 1.2

    def test_pushing_into_wall_counts_as_idle(self):
        env = AimTrainerEnv(
            max_steps=100,
            n_targets=1,
            target_radius=0.04,
            time_penalty=0.1,
        )
        env.reset(seed=0)
        env.crosshair = np.array([0.0, 0.5], dtype=np.float32)
        env.targets[0] = np.array([0.8, 0.5], dtype=np.float32)
        env.target_alive[0] = True

        _, reward, _, _, _ = env.step(np.array([-env.action_step, 0.0], dtype=np.float32))

        assert reward <= -0.6

    def test_step_info_contains_reward_breakdown(self):
        env = AimTrainerEnv(max_steps=100, n_targets=1, target_radius=0.04, lock_on_speed=0.0)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.array([env.action_step, 0.0], dtype=np.float32))

        assert "reward_total" in info
        assert "reward_progress" in info
        assert "reward_distance_delta" in info
        assert "reward_hit" in info
        assert "reward_time_penalty" in info
        assert "reward_idle_penalty" in info
        assert np.isclose(info["reward_total"], reward)

    def test_render_rgb_array(self):
        env = AimTrainerEnv(max_steps=10, n_targets=3, render_mode="rgb_array")
        env.reset()
        img = env.render()
        assert img is not None
        assert img.shape[2] >= 3  # RGB or RGBA
        env.close()
