from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


class Bandit():
    def __init__(self, kArms, trials, epsilon, trueReward, initial=0):
        self.k = kArms
        self.trials = trials
        self.epsilon = epsilon
        self.indices = np.arange(self.k)

        # real reward for each action
        self.qTrue = []
        # estimation for each actio
        self.qEst = np.zeros(self.k)
        self.num  = np.zeros(self.k)
        self.trueReward = trueReward
        self.Rewards = np.zeros(self.trials)

        # initialize real rewards with N(0,1) distribution and estimations with desired initial value
        for i in range(0, self.k):
            self.qTrue.append(np.random.randn() + trueReward[i])
            self.qEst[i] = initial

    def calculateReward(self):
        for step in range(self.trials):
            action = self.DecideOnAction()
            reward = self.PlayBandit(action)

            # update
            self.num[action] += 1
            self.qEst[action] += (reward-self.qEst[action]) / self.num[action]

            self.Rewards[step] = reward

        return self.Rewards

    def DecideOnAction(self):
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1:
                return np.random.choice(self.indices)

        return np.argmax(self.qEst)

    def PlayBandit(self, action):
        return np.random.randn() + self.qTrue[action]



def epsilonGreedy(kArms, trials, trueReward, epsilon=0.01):
    b = Bandit(kArms, trials, epsilon, trueReward)
    reward = b.calculateReward()
    print(reward)

    figureIndex = 1
    indices = np.arange(trials)
    plt.figure(figureIndex)
    plt.plot(indices, reward, label="reward")
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()


def main():
    epsilonGreedy(10, 1000, [1,2,3,4,5,6,7,8,9,10], 0.01)

# 直接実行された場合のみ実行し、それ以外の場合は実行しない」
if __name__ == '__main__':
    main()
