import argparse
import os 
import time
import gym
import gym_ple
import random
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from nn_rl import Net, Agent, Observer, ReplayBuffer, Trainer, Logger 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNet(Net):
    def __init__(self, actions):
        super(DQNNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*6*6, 256)
        self.fc2 = nn.Linear(256, len(actions))
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent(Agent):
    def __init__(self, env, epsilon, gamma, lr, model_path):
        super(DQNAgent, self).__init__(env, epsilon, gamma, lr, model_path)
        self.net = DQNNet(self.actions).to(device)
        self.target_net = DQNNet(self.actions).to(device)
        self.target_net.eval()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        #self.loss_fn = F.smooth_l1_loss

    def update_target_network(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def update(self, experiences):
        states = torch.stack([e.s for e in experiences])
        actions = torch.stack([e.a for e in experiences])
        rewards = torch.stack([e.r for e in experiences])
        n_states = torch.stack([e.n_s for e in experiences])

        # takes care for None state
        n_states_mask = torch.tensor(
                        tuple(map(lambda s: s is not None, n_states)),
                        device=device,
                        dtype=torch.bool)
        n_states = torch.stack([s for s in n_states if s is not None])

        # computes Q(s,a) w/ selected s,a
        # gather 1 == select column 1, then apply actions
        q = self.net(states).gather(1, actions)
        # computes Q'(n_s,:)
        q_next = torch.zeros(len(experiences), device=device)
        q_a_next = self.target_net(n_states)
        q_next[n_states_mask] = q_a_next.max(1)[0].detach()

        updated_q = rewards + (self.gamma*q_next.unsqueeze(1))

        loss = self.loss_fn(updated_q, q)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1,1)

        self.optimizer.step()

        #return loss

class CatcherObserver(Observer):
    def __init__(self, env, width, height, frame_count):
        super(CatcherObserver, self).__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)
    
    def transform(self, s):
        # mode L means gray scale
        grayed = Image.fromarray(s).convert(mode="L")
        # resize
        resized = grayed.resize((self.width, self.height))
        # convert to numpy float
        resized = np.array(resized).astype("float")
        # normalize
        normalized = resized / 255.0

        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)

        feature = np.array(self._frames)
        # Convert the feature shape (frames, width, height) -> (h,w,f)
        #feature = np.transpose(feature, (1,2,0))

        return torch.from_numpy(feature).float().to(device)

class DQNTrainer(Trainer):
    def __init__(self,
            env,
            init_eps=0.5,
            end_eps=0.001,
            buffer_size=10000, 
            batch_size=32, 
            gamma=0.99, 
            lr = 1e-4,
            episodes=1200,
            print_freq=10, 
            writer_freq=10, 
            target_network_freq=5,
            log_path='runs/dqn_pytorch',
            model_path='model/dqn_pytorch.h5'):
        super(DQNTrainer, self).__init__(
            env, init_eps, end_eps, buffer_size, batch_size, gamma, lr, 
            episodes, print_freq, writer_freq, target_network_freq,
            log_path)
        self.agent = DQNAgent(env, init_eps, gamma, lr, model_path)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.best_reward = -np.inf
    
    # we need env, agent, replay_buffer, logger
    def train(self, render=False, sleep=-1):
        # Start Rolling & Training at the same time
        for e in range(self.episodes):
            self.begin_episode()
            s = self.env.reset()
            done = False
            
            while not done:
                if render:
                    self.env.render()
                if sleep>0:
                    time.sleep(sleep)

                a, qvalue = self.agent.policy(s)
                n_s, r, done, info = self.env.step(a.item())

                self.replay_buffer.append(s, a, r, n_s, done)

                if self.replay_buffer.is_full():
                    batch = self.replay_buffer.sample()
                    self.agent.update(batch)
                    #self.loss += loss.item()

                self.reward += r.item()
                self.qvalue += qvalue
                self.step += 1

                s = n_s

            self.end_episode()

        return self.loss

    def begin_episode(self):
        self.loss = 0
        self.reward = 0
        self.qvalue = 0
        self.step = 0

    def end_episode(self):
        self.loss_log.append(self.loss)
        self.reward_log.append(self.reward)
        self.qvalue_log.append(self.qvalue)

        if _is_update(self.episode, self.print_freq):
            s = self.episode-self.print_freq
            reward = np.mean(self.reward_log[s:])
            self.logger.print('reward', reward, self.episode)
        
        if _is_update(self.episode, self.writer_freq):
            s = self.episode-self.writer_freq
            loss = np.mean(self.loss_log[s:])
            reward = np.mean(self.reward_log[s:])
            qvalue = np.mean(self.qvalue_log[s:])
            self.logger.write('loss', loss, self.episode)
            self.logger.write('reward', reward, self.episode)
            self.logger.write('qvalue', qvalue, self.episode)

        # update target network
        if _is_update(self.episode, self.target_network_freq):
            self.agent.update_target_network()

        # save model if it performed well
        if self.reward > self.best_reward:
            self.agent.save()
            self.best_reward = self.reward

        # update epsilon 
        self.agent.epsilon -= (self.init_eps-self.end_eps)/self.episodes
        # increment episode 
        self.episode += 1

def _is_update(episode, freq):
    if episode!=0 and episode%freq==0:
        return True
    return False


def main(args):
    env = gym.make(args.env_name)
    obs = CatcherObserver(env, 80, 80, 4)

    if args.play:
        model_path = os.path.join('model', 'dqn_pytorch.h5')
        agent = DQNAgent.load(obs, model_path)
        agent.play(render=True, sleep=args.sleep)
    else:
        trainer = DQNTrainer(obs)
        trainer.train(render=args.render, sleep=args.sleep)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN Continuous RL')
    parser.add_argument('--play', action='store_true', help="play with trained model")
    parser.add_argument('--render', action='store_true', help="play with trained model")
    parser.add_argument('--env_name', type=str, default='Catcher-v0',
                            help="Name of the Environment")
    parser.add_argument('--sleep', type=float, default=-1)
    args = parser.parse_args()
    main(args)