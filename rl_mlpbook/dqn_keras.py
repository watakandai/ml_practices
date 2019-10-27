"""
Algorithms:
    - DQN (Catcher/CartPole)
    - Policy Gradient (CartPole)
    - (A2C)
"""
import os
import argparse
import time
import gym
import gym_ple
import random
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt

class ReplayBuffer():
    def __init__(self, buffer_size=10000, batch_size=32):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences = deque(maxlen=buffer_size)

    def append(self, s, a, r, n_s, d):
        self.experiences.append({"s":s, "a":a, "r": r, "n_s":n_s, "d":d})

    def sample(self):
        return random.sample(self.experiences, self.batch_size)

    def is_full(self):
        return len(self.experiences) == self.buffer_size


class DQNAgent():
    def __init__(self, env, model_path, episode=200, gamma=0.99, learning_rate=1e-4, init_eps=0.5, final_eps=1e-3):
        self.env = env
        self.model_path = model_path
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.actions = list(range(env.action_space.n))
        self.model = self.define_model()
        self.optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)
        self._target_model = K.models.clone_model(self.model)
        self.model.compile(self.optimizer, loss='mse')
        self.epsilon = init_eps
        self.decay = max((init_eps - final_eps)/episode, 0) # no negative
        self.training = False
        self.loss = 0

    def define_model(self):
        normal = K.initializers.glorot_normal()

        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=self.env.reset().shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal, activation="relu"
        ))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer=normal, activation="relu"
        ))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(256, kernel_initializer=normal,
                                 activation="relu"))
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal))
        return model

    def reset(self):
        self.loss = 0

    def update_target_network(self):
        if self.training:
            self._target_model.set_weights(self.model.get_weights())

    def update_epsilon(self):
        if self.training:
            self.epsilon -= self.decay

    def train(self, experiences):
        if self.training is False:
            self.training = True
        
        states = np.array([e['s'] for e in experiences])
        n_states = np.array([e['n_s'] for e in experiences])

        q_a = self.model.predict(states)
        q_a_future = self._target_model.predict(n_states)

        for i, e in enumerate(experiences):
            reward = e['r']
            if not e['d']:
                reward += self.gamma * np.max(q_a_future[i])
            q_a[i][e['a']] = reward

        loss = self.model.train_on_batch(states, q_a)
        self.loss += loss

    def policy(self, s):
        # Initial Random Policy
        #if not self.training:
        #    return np.random.randint(len(self.actions))
        # Epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        # Greedy

        return np.argmax(self.model.predict(np.array([s]))[0])

    def save(self):
        self.model.save(self.model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path, init_eps=0.0001):
        agent = cls(env, model_path, init_eps=init_eps)
        agent.model = K.models.load_model(model_path)
        return agent

    def play(self, episode_count=10, render=True):
        for e in range(episode_count):
            s = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    self.env.render()
                a = self.policy(s)
                n_s, r, done, info = self.env.step(a)
                episode_reward += r 
                s = n_s
                #time.sleep(0.1)
            else:
                print("Get reward {}.".format(episode_reward))


class Observer():

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        raise NotImplementedError("You have to implement transform method.")

class CatcherObserver(Observer):

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # Convert the feature shape (f, w, h) => (h, w, f).
        feature = np.transpose(feature, (1, 2, 0))

        return feature

def train(env, model_path, render=False):
    episode = 1000
    target_network_update = 3

    sess = tf.InteractiveSession()
    agent = DQNAgent(env, model_path, episode=episode)
    replay_buffer = ReplayBuffer()

    log_dir = ".\\log"
    summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    with tf.Session() as sess:
        for e in range(episode):
            # Every other step, update target network
            if e % target_network_update == 0:
                agent.update_target_network()
            s = env.reset()
            agent.reset()

            done = False
            steps = 1

            while not done:
                if render:
                    env.render()
                # Choose Action
                a = agent.policy(s)
                # Play
                n_s, r, done, info = env.step(a)
                #time.sleep(0.05)
                # Add to buffer
                replay_buffer.append(s, a, r, n_s, done)

                if replay_buffer.is_full():
                    # Sample from buffer
                    batch = replay_buffer.sample()
                    # Train
                    agent.train(batch)
                    # Save loss in tensorboard
                
                steps += 1

                # Update state
                s = n_s

            summary = tf.Summary(value=[tf.Summary.Value(tag="loss", 
                                simple_value=agent.loss), ])
            summary_writer.add_summary(summary, e)

            agent.save()
            agent.update_epsilon()
            episode_rewards = np.array([ex['r'] for ex in replay_buffer.experiences])
            reward_sum = np.sum(episode_rewards[episode_rewards.size-steps:])
            print('Episode:%i, Steps:%i,   Reward:%.2f,   loss:%.2f, eps:%.2f'%(e, steps, reward_sum, agent.loss, agent.epsilon))

def parse_arguments():
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    parser.add_argument("--env_name", type=str, default='Catcher-v0',
                        help="name of the environment")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    save_dir = 'model'
    filename = "dqn_agent.h5"
    model_path = os.path.join(save_dir, filename)

    env = gym.make(args.env_name)
    obs = CatcherObserver(env, 80, 80, 4)

    if args.play:
        agent = DQNAgent.load(obs, model_path)
        agent.play(render=True)
    else:
        train(obs, model_path)
