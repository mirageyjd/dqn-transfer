import gym
from agent import Agent
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from skimage import transform

def scale_lumininance(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])


def state_preprocess(s):
    # convert into greyscale image
    _state = scale_lumininance(s)
    # Scale into 84*84
    _state = transform.resize(_state, (84, 84))
    _state = np.moveaxis(_state, 1, 0)
    return _state[np.newaxis, :]


def train_agent(env: gym.Env, agent: Agent, replay_buffer: ReplayBuffer, config: dict):
    s = env.reset()
    for t in range(1, config['t_max'] + 1):
        # sampling from environment
        epsilon = config['eps_start'] - (config['eps_start'] - config['eps_end']) * (t - 1) / (config['eps_end_t'] - 1)
        a = agent.action(torch.from_numpy(state_preprocess(s)).float().to(config['device']), epsilon)
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
                done = False
                eval_s = env.reset()
                # TODO: looger start
                for eval_t in range(config['eval_t']):
                    eval_a = agent.action(eval_s, config['eval_eps'])
                    eval_s2, eval_r, done, _ = env.step(eval_a)
                    # TODO: logger record eval_r
                    eval_s = env.reset() if done else eval_s2
                # TODO: looger end
                done = True

        s = env.reset() if done else s2
