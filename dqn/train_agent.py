from typing import Tuple
import gym
import torch
from dqn.agent import Agent
from dqn.replay_buffer import ReplayBuffer
from logger import Logger
from tqdm import tqdm


def train_agent(env: gym.Env, agent: Agent, replay_buffer: ReplayBuffer, logger: Logger, config: dict):
    start_t = 0
    if config['recover']:
        start_t = max(start_t, config['recover_t'] - config['learning_start'])
        config['learning_start'] += start_t
        agent.load_model_from_state_dict(torch.load(config['model_path']))
        logger.recover(config['recover_model_path'], config['recover_t'])
    elif config['pretrain']:
        agent.load_model_from_state_dict(torch.load(config['model_path']))
        logger.pretrain(config['pretrain_model_path'])

    s = env.reset()
    for t in tqdm(range(start_t + 1, config['t_max'] + 1)):
        # sampling from environment
        epsilon = config['eps_start'] - (config['eps_start'] - config['eps_end']) * (t - 1) / (
                    config['eps_end_t'] - 1) if t <= config['eps_end_t'] else config['eps_end']
        a = agent.action(s, epsilon)
        s2, r, done, _ = env.step(a)
        replay_buffer.insert((s, a, r, s2, done))

        if t > config['learning_start']:
            if t % config['update_freq'] == 0:
                # sample a mini-batch from replay buffer and update q function
                s_batch, a_batch, r_batch, s2_batch, done_batch = replay_buffer.sample(config['batch_size'])
                agent.train(s_batch, a_batch, r_batch, s2_batch, done_batch, config['gamma'])

            if t % config['target_update_freq'] == 0:
                # update target q function
                agent.update_target()

            if t % config['eval_freq'] == 0:
                avg_reward, num_episode = eval_agent(env, agent, config)
                logger.record(t, avg_reward, num_episode)
                done = True

            if t % config['checkpoint_freq'] == 0:
                logger.save_model(agent.get_model())

        s = env.reset() if done else s2

    logger.train_over()


def eval_agent(env: gym.Env, agent: Agent, config: dict) -> Tuple[float, int]:
    total_reward = 0.0
    num_episode = 0

    s = env.reset()
    done = False
    for t in range(config['eval_t']):
        a = agent.action(s, config['eval_eps'])
        s2, r, done, _ = env.step(a)
        total_reward += r

        if done:
            s = env.reset()
            num_episode += 1
        else:
            s = s2

    if not done:
        if config['eval_complete_episode']:
            while not done:
                a = agent.action(s, config['eval_eps'])
                s2, r, done, _ = env.step(a)
                total_reward += r
                s = s2

        num_episode += 1

    return total_reward / num_episode, num_episode
