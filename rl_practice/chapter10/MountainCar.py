from __future__ import print_function
import time
from math import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MountainCar:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.x = 0.0
        self.xdot = 0.0
        self.isGoal = False

    def play(self):
        curr_state = (np.random.rand()-0.6, 0.0)
        self.reset(curr_state)
        while self.isGoal is False:
            # choose Action from policy
            curr_action = self.take_action()
            # update state, reward
            next_state, next_reward = self.update(curr_action)
            # save
            self.states.append(curr_state)
            self.actions.append(curr_action)
            self.rewards.append(next_reward)
            curr_state = next_state

        return self.states, self.actions, self.rewards

    @staticmethod
    def take_action():
        return np.random.choice([-1, 0, 1])

    # action = -1, 0, 1
    # Vt+1 = bound[Vt + 0.001*At - 0.0025*cos(3Xt)]
    # Xt+1 = bound[Xt + Vt+1]
    def update(self, action):
        # Velocity
        self.xdot = self.xdot + 0.001*action - 0.0025*cos(3*self.x)
        if self.xdot > 0.07:
            self.xdot = 0.07
        elif self.xdot < -0.07:
            self.xdot = -0.07
        # Position
        self.x = self.x + self.xdot
        # reward
        if self.x >= 0.5:
            self.isGoal = True
        elif self.x <= -1.2:
            self.x = np.random.rand()-0.6
            self.xdot = 0.0

        return (self.x, self.xdot), -1

    def reset(self, state):
        self.states = []
        self.actions = []
        self.rewards = []
        self.x = state[0]
        self.xdot = state[1]
        self.isGoal = False


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return np.random.sample(self._samples, len(self._samples))
        else:
            return np.random.sample(self._samples, no_samples)


# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
figure_index = 0


# print learned cost to go
def pretty_print(state_action_value, title):
    gridSize = 40
    positionStep = (POSITION_MAX - POSITION_MIN) / gridSize
    positions = np.arange(POSITION_MIN, POSITION_MAX + positionStep, positionStep)
    velocityStep = (VELOCITY_MAX - VELOCITY_MIN) / gridSize
    velocities = np.arange(VELOCITY_MIN, VELOCITY_MAX + velocityStep, velocityStep)
    axisX = []
    axisY = []
    axisZ = []
    for position in positions:
        for velocity in velocities:
            axisX.append(position)
            axisY.append(velocity)
            axisZ.append(state_action_value(position, velocity))

    fig = plt.figure(figure_index)
    figure_index += 1
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(axisX, axisY, axisZ)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')


if __name__ == '__main__':
    nEpisode = 12
    episodes = []
    mountain_car = MountainCar()
    for i in range(nEpisode):
        s, a, r = mountain_car.play()
        episodes.append([s, a, r])
        print('Episode: {}th'.format(i))
        print('{} Steps'.format(len(s)))
        time.sleep(0.5)

    nql = NeuralQLearning()
    for i in range(nEpisode):
        nql.learn(episodes[i])

    # prettyPrint()
    plt.show()
