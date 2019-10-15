"""
Algorithms:
    - DQN (Catcher/CartPole)
    - Policy Gradient (CartPole)
    - (A2C)
"""
import os
import argparse
import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from collections import deque


class ReplayBuffer():
    def __init__(self, buffer_size=50000, batch_size=32):
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
    def __init__(self, env, model_path, episode=1000, learning_rate=1e-3, init_eps=0.5, final_eps=1e-3):
        self.env = env
        self.model_path = model_path
        self.learning_rate = self.learning_rate
        self.actions = list(range(env.action_space.n))
        self.model = self.define_model()
        self.optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)
        self.model.compile(self.optimizer, loss='mse')
        self._target_model = K.models.clone_model(self.model)
        self.epsilon = init_eps
        self.decay = max((initial_epsilon - final_epsilon)/episode, 0) # no negative
        self.training = False

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
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal,
                                 activation="relu"))
        return model

    def update_target_network():
        if self.training:
            self._target_model.set_weights(self.model.get_weights())

    def update_epsilon():
        if self.training:
            self.epsilon -= self.decay

    def train(self, batch):
        if self.training is False:
            self.training = True

        x_batch = []
        y_batch = []
        for sample in batch:
            # from experiences
            s = sample['s']
            a = sample['a']
            n_s = sample['n']
            done = sample['d']

            curr_q = self.model.predict(s)
            stable_q = self._target_model.predict(n_s)

            r = e['r']
            if not done:
                r += gamma * np.max(stable_q) - curr_q[a]
            curr_q[a] = r

            x_batch.append(s)
            y_batch.append(curr_q)

        loss = self.model.train_on_batch(x_batch, y_batch)
        return loss

    def policy(self, s):
        # Initial Random Policy
        if not self.training:
            return np.random.randint(len(self.actions))
        # Epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        # Greedy
        return np.argmax(self.model.predict(s))

    def save(self):
        self.model.save(self.model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path, init_eps=0.0001):
        agent = cls(env, model_path, init_eps)
        agent.model = K.models.load_model(model_path)
        return agent

    def play(self, episode_count=5, render=True):
        for e in range(episode_count):
            s = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    self.env.render()
                a = self.policy(s)
                n_s, r, done, info = self.env.step(a)
                episode_reward += reward
                s = n_s
            else:
                print("Get reward {}.".format(episode_reward))


def train(env, model_path):
    episode = 1000
    target_network_update = 10

    sess = tf.InteractiveSession()
    agent = Agent(env, model_path, episode=episode)
    replay_buffer = ReplayBuffer()

    log_dir = ".\\log"
    summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    with tf.Session() as sess:
        for e in range(episode):
            # Every other step, update target network
            if e % target_network_update == 0:
                agent.update_target_network()

            done = False
            while not done:
                # Choose Action
                a = agent.policy(s)
                # Play
                n_s, r, done, info = agent.play(a)
                # Add to buffer
                replay_buffer.append(s, a, r, n_s, done)

                if replay_buffer.is_full():
                    # Sample from buffer
                    batch = replay_buffer.sample()
                    # Train
                    loss = agent.train(batch)
                    # Save loss in tensorboard
                    # TODO
                    summary = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                                simple_value=loss), ])
                    summary_writer.add_summary(summary)

                # Update state
                s = n_s

            agent.save(file_name)
            agent.update_epsilon()

def parse_arguments():
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    parser.add_argument("--env_name", type=str, default='CartPole-v0',
                        help="name of the environment")

    return parser.parse_args()

def __name__ == '__main__':
    args = parse_arguments()

    save_dir = 'model'
    filename = "dqn_agent.h5"
    model_path = os.path.join(save_dir, filename)

    env = gym.make(args.env_name)

    if args.play:
        agent = DQNAgent.load(env, model_path)
        agent.play()
    else:
        train(env, model_path)


