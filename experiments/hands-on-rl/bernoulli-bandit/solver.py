import torch
import random

class Solver:
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = torch.zeros(size=(bandit.K,))  # 记录每个拉杆被拉动的次数
        self.regret = 0.
        self.regrets = []
        self.actions = []
    def update_regret(self,action):
        self.regret += (self.bandit.best_prob - self.bandit.probs[action]).item()
        self.regrets.append(self.regret)
    def select_action(self):
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
            
class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = torch.tensor([init_prob] * self.bandit.K)

    def run_one_step(self):
        if random.random() < self.epsilon:
            k = random.randint(0, self.bandit.K - 1)  # 随机选择一根拉杆
        else:
            k = torch.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        
        self.total_count = 0
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = torch.tensor([init_prob] * self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        if random.random() < self.epsilon / self.total_count:
            k = random.randint(0, self.bandit.K - 1)  # 随机选择一根拉杆
        else:
            k = torch.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

