import gym
from network import q_network_atari_creator
from agent import Agent
from replay_buffer import ReplayBuffer
from train_agent import train_agent

config = {
    'device': 'cpu',
    'log_dir': '',

    'env_name': 'PongNoFrameskip-v4',
    'q_network_creator': q_network_atari_creator,

    # deepmind-style atari (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    'frame_skip': 4,                # frame skip
    'action_repeat': 4,             # number of actions repeated after the agent takes each action
    'history_len': 4,               # agent history length
    'no_op_max': 30,                # maximum number of no-op at the start of each episode

    # training hyperparameters (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    't_max': 50000000,              # maximum training steps(frames)
    'learning_start': 50000,        # number of steps before learning starts
    'replay_capacity': 1000000,     # replay buffer size
    'target_update_freq': 10000,    # target network update frequency
    'update_freq': 4,               # update frequency between successive SGD
    'gamma': 0.99,                  # discount factor
    'batch_size': 32,               # minibatch size

    # linearly-annealed epsilon-greedy (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    'eps_start': 1.0,  # initial epsilon value
    'eps_end': 0.1,  # final epsilon value
    'eps_end_t': 1000000,  # the step(frame) that epsilon reaches final epsilon value

    # Adam optimizer (from https://arxiv.org/pdf/1710.02298.pdf)
    'adam_lr': 0.0000625,           # learning rate
    'adam_eps': 0.00015,            # epsilon

    # evaluation hyperparameters (from https://github.com/deepmind/dqn)
    'eval_freq': 250000,            # evaluation frequency
    'eval_t': 125000,               # number of steps(frames) in evaluation
    'eval_eps': 0.05,               # value for epsilon-greedy in evaluation
}

env = gym.make(config['env_name'])
# TODO: wrap atari game according to config
agent = Agent(env, config)
replay_buffer = ReplayBuffer(env, config)
# TODO: implement logger
train_agent(env, agent, replay_buffer, config)
