"""悬崖漫步（CliffWalking）环境实现。

4×12 网格世界：
- 起点 (3, 0)，终点 (3, 11)
- 底部中间一行是悬崖，掉入回到起点并受到 -100 惩罚
- 每步 -1 奖励，到达终点结束

动作空间（4 个离散动作）：
    0: 上, 1: 右, 2: 下, 3: 左
"""

import numpy as np

# ── 网格常量 ──────────────────────────────────────────────
ROWS = 4
COLS = 12
N_STATES = ROWS * COLS  # 48
N_ACTIONS = 4

# 起点与终点
START = (3, 0)
GOAL = (3, 11)

# 悬崖：底部中间一整行
CLIFF = [(3, c) for c in range(1, 11)]  # (3,1) ~ (3,10)


class CliffWalkingEnv:
    """悬崖漫步环境。

    提供类 Gymnasium 的 reset / step / render 接口，
    纯 NumPy 实现，无外部依赖（除 numpy）。
    """

    def __init__(self) -> None:
        self.rows = ROWS
        self.cols = COLS
        self.n_states = N_STATES
        self.n_actions = N_ACTIONS

        # 当前智能体位置
        self._agent_pos: tuple[int, int] = START

    # ── 核心接口 ──────────────────────────────────────────

    def reset(self) -> int:
        """重置环境到起点，返回初始状态编号。"""
        self._agent_pos = START
        return self._pos_to_state(START)

    def step(self, action: int) -> tuple[int, float, bool]:
        """执行动作，返回 (下一状态, 奖励, 是否终止)。

        Parameters
        ----------
        action : int
            0=上, 1=右, 2=下, 3=左。

        Returns
        -------
        next_state : int
            转移后状态编号 (0~47)。
        reward : float
            即时奖励。
        done : bool
            是否到达终止状态（终点或悬崖）。
        """
        row, col = self._agent_pos

        # 计算目标位置（边界外则保持不动）
        if action == 0:  # 上
            next_row = max(row - 1, 0)
            next_col = col
        elif action == 1:  # 右
            next_row = row
            next_col = min(col + 1, self.cols - 1)
        elif action == 2:  # 下
            next_row = min(row + 1, self.rows - 1)
            next_col = col
        elif action == 3:  # 左
            next_row = row
            next_col = max(col - 1, 0)
        else:
            raise ValueError(f"无效动作: {action}，取值范围 0~3")

        next_pos = (next_row, next_col)

        # 奖励与终止判断
        if next_pos == GOAL:
            self._agent_pos = next_pos
            return self._pos_to_state(next_pos), -1.0, True

        if next_pos in CLIFF:
            # 掉下悬崖：回到起点，惩罚 -100
            self._agent_pos = START
            return self._pos_to_state(START), -100.0, True

        # 普通移动
        self._agent_pos = next_pos
        return self._pos_to_state(next_pos), -1.0, False

    # ── 渲染 ──────────────────────────────────────────────

    def render(self) -> None:
        """在终端打印当前网格状态。"""
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]

        for r, c in CLIFF:
            grid[r][c] = "C"
        grid[START[0]][START[1]] = "S"
        grid[GOAL[0]][GOAL[1]] = "G"

        ar, ac = self._agent_pos
        if (ar, ac) not in (START, GOAL):
            grid[ar][ac] = "A"
        elif (ar, ac) == START and (ar, ac) != GOAL:
            grid[ar][ac] = "A"  # 智能体在起点时显示 A

        print("-" * (self.cols + 2))
        for r in range(self.rows):
            row_str = "".join(grid[r])
            print(f"|{row_str}|")
        print("-" * (self.cols + 2))
        print(f"位置: ({ar}, {ac})  状态: {self._pos_to_state((ar, ac))}\n")

    # ── 工具方法 ──────────────────────────────────────────

    def state_to_pos(self, state: int) -> tuple[int, int]:
        """状态编号 → 网格坐标。"""
        return divmod(state, self.cols)

    def _pos_to_state(self, pos: tuple[int, int]) -> int:
        """网格坐标 → 状态编号。"""
        return pos[0] * self.cols + pos[1]


# ── 交互式测试入口 ────────────────────────────────────────


def main() -> None:
    """命令行交互：用键盘控制 Agent，直接感受环境。"""
    print("=" * 50)
    print("悬崖漫步 (CliffWalking) 环境 - 键盘试玩")
    print("=" * 50)
    print("动作: W=上  D=右  S=下  A=左  Q=退出  R=重置\n")

    env = CliffWalkingEnv()
    state = env.reset()
    env.render()

    action_map = {"w": 0, "d": 1, "s": 2, "a": 3}

    total_reward = 0
    step_count = 0

    while True:
        key = input(">> ").strip().lower()

        if key == "q":
            print("退出。")
            break
        if key == "r":
            state = env.reset()
            total_reward = 0
            step_count = 0
            env.render()
            continue
        if key not in action_map:
            print("无效按键，请用 W/A/S/D")
            continue

        action = action_map[key]
        state, reward, done = env.step(action)
        total_reward += reward
        step_count += 1
        env.render()

        print(f"步数: {step_count}  奖励: {reward:.0f}  累计: {total_reward:.0f}")

        if done:
            if state == env._pos_to_state(GOAL):
                print("🎉 到达终点！")
            else:
                print("💀 掉下悬崖！")
            print(f"总步数: {step_count}，总奖励: {total_reward:.0f}")
            # 自动重置
            input("按回车继续下一轮...")
            state = env.reset()
            total_reward = 0
            step_count = 0
            env.render()
