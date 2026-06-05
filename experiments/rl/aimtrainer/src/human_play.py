"""人类可玩 Aim Trainer — 用鼠标瞄准并点击射击

与 RL agent 对比：同样的环境、同样的规则，看看你的手速有多快
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame

from env.aim_trainer_env import AimTrainerEnv

WINDOW_SIZE = 600
TARGET_RADIUS = 24  # 人类友好的命中半径
N_TARGETS = 5
MAX_STEPS = 500  # 约 15 秒
FPS = 30

# 颜色
BG = (15, 15, 45)
GRID = (30, 30, 70)
TARGET = (255, 60, 60)
CURSOR = (83, 168, 182)
HIT_FLASH = (255, 215, 0)
TEXT = (220, 220, 220)


def play():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Aim Trainer — Human Play")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui,Arial,PingFangSC", 18)
    font_big = pygame.font.SysFont("segoeui,Arial,PingFangSC", 28)

    # 初始化环境
    env = AimTrainerEnv(
        max_steps=MAX_STEPS,
        n_targets=N_TARGETS,
        target_radius=TARGET_RADIUS / WINDOW_SIZE,
    )
    obs, _ = env.reset()

    running = True
    pause = False
    last_hit_frame = 0  # 用于命中 flash 效果

    while running:
        dt = clock.tick(FPS)
        current_step = env.current_step

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # 空格重置
                    obs, _ = env.reset()
                    last_hit_frame = 0
                elif event.key == pygame.K_p:
                    pause = not pause
            elif event.type == pygame.MOUSEBUTTONDOWN and not pause:
                # 鼠标点击：将像素坐标转为归一化坐标
                mx, my = pygame.mouse.get_pos()
                nx = mx / WINDOW_SIZE
                ny = my / WINDOW_SIZE

                # 检查是否命中任何目标
                for i in range(env.n_targets):
                    if not env.target_alive[i]:
                        continue
                    dist = np.sqrt(
                        (nx - env.targets[i][0]) ** 2 + (ny - env.targets[i][1]) ** 2
                    )
                    if dist < env.target_radius:
                        # 命中！
                        env.total_hits += 1
                        env.target_alive[i] = False
                        env._spawn_target(i)
                        last_hit_frame = current_step
                        # 更新观测
                        obs = env._get_obs()
                        break

        if not pause and running:
            # 自动步进（时间惩罚）
            env.current_step += 1
            obs = env._get_obs()

            if env.current_step >= MAX_STEPS:
                pause = True
                print(f"⏰ 时间到！命中: {env.total_hits}")

        # ── 渲染 ──
        screen.fill(BG)

        # 网格
        for i in range(0, WINDOW_SIZE, 40):
            pygame.draw.line(screen, GRID, (i, 0), (i, WINDOW_SIZE))
            pygame.draw.line(screen, GRID, (0, i), (WINDOW_SIZE, i))

        # 目标
        for i in range(env.n_targets):
            if not env.target_alive[i]:
                continue
            tx = int(env.targets[i][0] * WINDOW_SIZE)
            ty = int(env.targets[i][1] * WINDOW_SIZE)
            r = TARGET_RADIUS

            for glow in range(r + 10, r - 2, -2):
                c = (255, max(0, 120 - glow * 8), max(0, 120 - glow * 8))
                pygame.draw.circle(screen, c, (tx, ty), max(1, glow))
            pygame.draw.circle(screen, TARGET, (tx, ty), r, 2)
            pygame.draw.circle(screen, TARGET, (tx, ty), max(1, r // 2), 1)

        # 命中闪光
        if current_step - last_hit_frame < 10:
            flash = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            flash.set_alpha(max(0, 80 - (current_step - last_hit_frame) * 8))
            flash.fill(HIT_FLASH)
            screen.blit(flash, (0, 0))

        # 鼠标准星
        mx_px, my_px = pygame.mouse.get_pos()
        sz = 14
        pygame.draw.line(screen, CURSOR, (mx_px - sz, my_px), (mx_px + sz, my_px), 2)
        pygame.draw.line(screen, CURSOR, (mx_px, my_px - sz), (mx_px, my_px + sz), 2)
        pygame.draw.circle(screen, CURSOR, (mx_px, my_px), TARGET_RADIUS, 1)

        # HUD
        hits_s = font.render(f"Hits: {env.total_hits}", True, TEXT)
        time_s = font.render(f"Time: {current_step}/{MAX_STEPS}", True, TEXT)
        screen.blit(hits_s, (10, 10))
        screen.blit(time_s, (10, 36))

        if pause:
            pause_text = font_big.render("PAUSED (P to resume)", True, TEXT)
            screen.blit(pause_text, (WINDOW_SIZE // 2 - 150, WINDOW_SIZE // 2 - 20))

        # 指令提示
        hint = font.render("Click | Space=Reset | P=Pause", True, (100, 100, 100))
        screen.blit(hint, (10, WINDOW_SIZE - 28))

        pygame.display.flip()

    pygame.quit()
    print(f"\n🏆 最终命中: {env.total_hits}")


if __name__ == "__main__":
    play()
