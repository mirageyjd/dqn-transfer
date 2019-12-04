import gym
from network import q_network_atari_creator
from agent import Agent
from replay_buffer import ReplayBuffer
from train_agent import train_agent
from logger import Logger
from atari_env import AtariEnv, AtariTennisWrapper

config = {
    'device': 'cuda',
    'log_dir': './results/',
    'log_stdout': False,

    'experiment_name': 'tennis-1',
    'env_name': 'TennisNoFrameskip-v4',
    'q_network_creator': q_network_atari_creator,

    # action mapping: restrict action space
    'action_mapping_on': True,
    'action_mapping': ['FIRE', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE',
                       'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'],

    # deepmind-style atari (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    'frame_skip': 4,                 # frame skip
    'action_repeat': 4,              # number of actions repeated after the agent takes each action
    'history_len': 4,                # agent history length
    'no_op_max': 30,                 # maximum number of no-op at the start of each episode

    # training hyperparameters (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    't_max': 50000000,               # maximum training steps(frames)
    'learning_start': 50000,         # number of steps before learning starts
    'replay_capacity': 1000000,      # replay buffer size
    'target_update_freq': 10000,     # target network update frequency
    'update_freq': 4,                # update frequency between successive SGD
    'gamma': 0.99,                   # discount factor
    'batch_size': 32,                # minibatch size

    # linearly-annealed epsilon-greedy (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    'eps_start': 1.0,                # initial epsilon value
    'eps_end': 0.1,                  # final epsilon value
    'eps_end_t': 1000,            # the step(frame) that epsilon reaches final epsilon value

    # Adam optimizer (from https://arxiv.org/pdf/1710.02298.pdf)
    'adam_lr': 0.0000625,            # learning rate
    'adam_eps': 0.00015,             # epsilon

    # evaluation hyperparameters (from https://github.com/deepmind/dqn)
    'eval_freq': 250000,             # evaluation frequency
    'eval_t': 125000,                # number of steps(frames) in evaluation
    'eval_eps': 0.05,                # value for epsilon-greedy in evaluation
    'eval_complete_episode': False,  # complete episode even if number of evaluation steps exceeds

    'checkpoint_freq': 1000000,      # checkpoint for saving model

    # recover training
    'recover': False,
    'recover_t': 0,
    'model_path': './results/tennis-1.model',
}

env = AtariTennisWrapper(gym.make(config['env_name']))
env = AtariEnv(env, config)
agent = Agent(env, config)
replay_buffer = ReplayBuffer(env, config)
logger = Logger(config)
train_agent(env, agent, replay_buffer, logger, config)
