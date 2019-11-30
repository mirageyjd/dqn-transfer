from typing import Tuple
import gym
from agent import Agent
from replay_buffer import ReplayBuffer
from logger import Logger
from tqdm import tqdm


def train_agent(env: gym.Env, agent: Agent, replay_buffer: ReplayBuffer, logger: Logger, config: dict):
    s = env.reset()
    for t in tqdm(range(1, config['t_max'] + 1)):
        # sampling from environment
        epsilon = config['eps_start'] - (config['eps_start'] - config['eps_end']) * (t - 1) / (config['eps_end_t'] - 1)
        a = agent.action(s, epsilon)
        s2, r, done, _ = env.step(a)
        replay_buffer.insert((s, a, r, s2))

        if t > config['learning_start']:
            if t % config['update_freq'] == 0:
                # sample a mini-batch from replay buffer and update q function
                s_batch, a_batch, r_batch, s2_batch = replay_buffer.sample(config['batch_size'])
                agent.train(s_batch, a_batch, r_batch, s2_batch, config['gamma'])

            if t % config['target_update_freq'] == 0:
                # update target q function
                agent.update_target()

            if t % config['eval_freq'] == 0:
                total_reward, num_episode = eval_agent(env, agent, config)
                logger.record(t, total_reward / num_episode, num_episode)
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
        num_episode += 1

    return total_reward, num_episode


