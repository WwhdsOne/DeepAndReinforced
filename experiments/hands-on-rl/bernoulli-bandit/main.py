import torch
import matplotlib.pyplot as plt
import random
from solver import EpsilonGreedy,DecayingEpsilonGreedy

class BernoulliBandit:
    """伯努利多臂老虎机,输入K表示拉杆个数"""

    def __init__(self, K):
        self.probs = torch.rand(size=(K,))  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = torch.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if random.random() < self.probs[k]:
            return 1
        else:
            return 0

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    torch.random.manual_seed(1)  # 设定随机种子,使实验具有可重复性
    random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print(
        "获奖概率最大的拉杆为%d号,其获奖概率为%.4f"
        % (bandit_10_arm.best_idx, bandit_10_arm.best_prob)
    )

    num_steps = 5000

    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(num_steps)
    print('epsilon-贪婪算法的累积懊悔为：%.4f' % epsilon_greedy_solver.regret)

    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm, epsilon=0.01)
    decaying_epsilon_greedy_solver.run(num_steps)
    print('decaying-epsilon-贪婪算法的累积懊悔为：%.4f' % decaying_epsilon_greedy_solver.regret)

    plot_results(
        [epsilon_greedy_solver, decaying_epsilon_greedy_solver],
        ["EpsilonGreedy (eps=0.01)", "DecayingEpsilonGreedy (eps=0.01/t)"],
    )

