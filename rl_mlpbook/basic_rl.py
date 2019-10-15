import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gym
import math

class BasicAgent():
    def __init__(self, env, epsilon=0.1, gamma=0.99):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = list(range(env.action_space.n))

    def policy(self, s):
        return np.random.randint(len(self.actions))

class MonteCarloAgent(BasicAgent):
    def __init__(self, env, epsilon=0.03, gamma=0.9):
        super().__init__(env, epsilon, gamma)
        self.Q = defaultdict(lambda: [0]*len(self.actions))
        self.N = defaultdict(lambda: [0]*len(self.actions))

    def policy(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))

        if s in self.Q and sum(self.Q[s]) != 0:
            return np.argmax(self.Q[s])
        else:
            return np.random.randint(len(self.actions))

    def train(self, epochs=1000, render=False):
        # for loop
        total_rewards = []
        for i_epoch in range(epochs):
            s = self.env.reset()

            experiences = []
            rewards = 0
            done = False

            while not done:
                if render:
                    self.env.render(mode="human")
                a = self.policy(s)
                n_state, reward, done, info = self.env.step(a)

                experiences.append({"state":s, "action":a, "reward": reward, "next_state":n_state})
                rewards += reward
                s = n_state

            total_rewards.append(rewards)

            for i, x in enumerate(experiences):
                s, a = x["state"], x["action"]

                discounted_reward = 0
                t = 0
                for j in range(i, len(experiences)):
                    reward = experiences[j]["reward"]
                    discounted_reward += np.power(self.gamma, 1)*reward
                    t += 1

                self.N[s][a] += 1
                alpha = 1 / self.N[s][a]
                self.Q[s][a] += alpha * (discounted_reward - self.Q[s][a])

        return total_rewards

class QLearningAgent(BasicAgent):
    def __init__(self, env, epsilon=0.03, gamma=0.9, alpha=0.1):
        super().__init__(env, epsilon, gamma)
        self.alpha = alpha
        self.Q = defaultdict(lambda: [0]*len(self.actions))
        self.N = defaultdict(lambda: [0]*len(self.actions))

    def policy(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))

        if s in self.Q and sum(self.Q[s]) != 0:
            return np.argmax(self.Q[s])
        else:
            return np.random.randint(len(self.actions))

    def train(self, epochs=1000, render=False):
        # for loop
        total_rewards = []
        for i_epoch in range(epochs):
            s = self.env.reset()

            experiences = []
            rewards = 0
            done = False

            while not done:
                if render:
                    self.env.render(mode="human")
                a = self.policy(s)
                n_state, reward, done, info = self.env.step(a)

                experiences.append({"state":s, "action":a, "reward": reward, "next_state":n_state})
                self.Q[s][a] += self.alpha * (reward + self.gamma*max(self.Q[n_state])  - self.Q[s][a])
                rewards += reward
                s = n_state

            total_rewards.append(rewards)

        return total_rewards

if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")
    # agent = MonteCarloAgent(env)
    agent = QLearningAgent(env)
    rewards = agent.train(epochs=10000, render=False)

    steps = 50
    mean_rewards = np.array([np.mean(rewards[max(0,i-steps):i]) for i in range(len(rewards))])
    std_rewards = np.array([np.std(rewards[max(0,i-steps):i]) for i in range(len(rewards))])
    plt.fill_between(range(len(rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.1, color='b')
    plt.plot(range(len(rewards)), mean_rewards)
    plt.show()
