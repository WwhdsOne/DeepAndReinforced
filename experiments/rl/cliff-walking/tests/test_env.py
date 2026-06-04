"""悬崖漫步环境单元测试。"""

import numpy as np
from cliff_walking import CliffWalkingEnv


class TestCliffWalkingEnv:
    """环境基本功能测试。"""

    def setup_method(self) -> None:
        self.env = CliffWalkingEnv()

    def test_reset(self) -> None:
        state = self.env.reset()
        assert state == 36  # (3, 0)
        assert self.env._agent_pos == (3, 0)

    def test_state_pos_conversion(self) -> None:
        assert self.env._pos_to_state((0, 0)) == 0
        assert self.env._pos_to_state((3, 11)) == 47
        assert self.env.state_to_pos(36) == (3, 0)

    def test_step_up(self) -> None:
        self.env.reset()
        for _ in range(3):  # 走到右上角附近
            self.env.step(0)  # 上
        state, reward, done = self.env.step(1)  # 右
        assert self.env._agent_pos == (0, 1)

    def test_step_boundary(self) -> None:
        """测试边界：在最左列向左不动、在最上行向上不动。"""
        # 从起点往左（应该不动）
        self.env.reset()
        state, _, _ = self.env.step(3)  # 左
        assert self.env._agent_pos == (3, 0)

    def test_fall_off_cliff(self) -> None:
        """测试掉下悬崖：应回到起点并返回 -100 奖励。"""
        self.env.reset()
        self.env.step(0)  # 上 → (2, 0)
        self.env.step(0)  # 上 → (1, 0)
        self.env.step(0)  # 上 → (0, 0)
        self.env.step(1)  # 右 → (0, 1)
        self.env.step(2)  # 下 → (1, 1)
        self.env.step(2)  # 下 → (2, 1)
        # 再往下一步到 (3, 1)，是悬崖
        state, reward, done = self.env.step(2)
        assert reward == -100.0
        assert done is True
        assert state == 36  # 回到起点

    def test_reach_goal(self) -> None:
        """测试到达终点。"""
        self.env.reset()
        # 从 (3,0) 往上到 (0,0)，往右到 (0,11)，往下到 (3,11)
        for _ in range(3):
            self.env.step(0)  # (0, 0)
        for _ in range(11):
            self.env.step(1)  # (0, 11)
        for _ in range(2):
            _, _, _ = self.env.step(2)  # (0,11) → (1,11) → (2,11)
        state, reward, done = self.env.step(2)  # (2,11) → (3,11) 即终点
        assert state == 47
        assert reward == -1.0
        assert done is True

    def test_random_rollout(self) -> None:
        """随机策略跑 100 轮，确保不会崩溃。"""
        for _ in range(100):
            self.env.reset()
            done = False
            steps = 0
            while not done and steps < 200:
                action = np.random.randint(0, 4)
                _, _, done = self.env.step(action)
                steps += 1
