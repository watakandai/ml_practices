import os
import sys
import time
import argparse
import gym
import numpy as np
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger, _is_update
from models.td3 import TD3Agent


def run_evaluation(args):
    env = gym.make(args.env) 
    
    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    scale = max_action * np.ones(action_dim)

    agent = TD3Agent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        scale=scale, 
        model_path='_'.join([args.model_path, args.date]))
    agent.load(args.timestep)

    rewards = agent.evaluate_policy(env, args.eval_episodes, args.render, args.save_video, args.sleep)
    mean = np.mean(rewards)
    std = np.std(rewards)
    median = np.median(rewards)

    print('mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}'.format(
        mean=mean, std=std, median=median))

    env.close()

def run_training(args):
    env = gym.make(args.env) 
    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    
    print('States: %i'%(state_dim))
    print('Actions: %i'%(action_dim))

    # TODO
    scale = max_action * np.ones(action_dim)

    agent = TD3Agent(
        state_dim, 
        action_dim, 
        scale, 
        '_'.join([args.model_path, args.date])
        )

    replay_buffer = ReplayBuffer(
        state_dim, 
        action_dim, 
        buffer_size=args.buffer_size,
        batch_size=args.batch_size
        )

    logger = Logger(log_path='_'.join([args.log_path, args.date]))

    run_train_loop(args, env, agent, replay_buffer, logger)

def run_train_loop(args, env, agent, replay_buffer, logger):
    accum_steps = 1

    for e in np.arange(args.episode)+1:
        s = env.reset()
        done = False
        steps = 1
        episode_reward = 0

        while not done:
            if accum_steps < args.start_training:
                a = env.action_space.sample()
            else:
                a = agent.policy_with_noise(s)

            n_s, r, done, _ = env.step(a)
            d = float(done) if steps < env._max_episode_steps else 0

            replay_buffer.append(s, a, n_s, r, d)

            if e > args.start_training:
                losses = agent.train(replay_buffer)

                if _is_update(steps, args.writer_freq):
                    for k, v in losses.items():
                        logger.write('loss/%s'%(k), v, accum_steps)

            s = n_s
            episode_reward += r
            steps += 1
            accum_steps += 1

        logger.write('reward', episode_reward, e)

        if _is_update(e, args.model_save_freq):
            agent.save(timestep=accum_steps-1)
            rewards = agent.evaluate_policy(env, args.eval_episodes)
            mean = np.mean(rewards)
            std = np.std(rewards)
            median = np.median(rewards)

            print('mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}'.format(
                mean=mean, std=std, median=median))

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--sleep', type=float, default=-1)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--eval_episodes', type=float, default=5)

    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--episode', default=1000, type=int)
    parser.add_argument('--buffer_size', default=int(1e6), type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--model_path', default='model/td3_updated', type=str)
    parser.add_argument('--log_path', default='log/td3_updated', type=str)
    parser.add_argument('--start_training', default=5, type=int)
    parser.add_argument('--writer_freq', default=10, type=int)
    parser.add_argument('--actor_lr', default=0.0001, type=int)
    parser.add_argument('--critic_lr', default=0.0001, type=int)
    parser.add_argument('--expl_noise', default=0.1, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=int)
    parser.add_argument('--noise_clip', default=0.5, type=int)
    parser.add_argument('--gamma', default=0.99, type=int)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--tau', default=0.005, type=int)
    parser.add_argument('--timestep', default=-1, type=int)

    parser.add_argument('--model_save_freq', default=500, type=int)

    args = parser.parse_args()

    if args.train:
        run_training(args)

    if args.eval:
        run_evaluation(args) 
