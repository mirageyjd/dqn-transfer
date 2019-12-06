import gym
from network import q_network_mlp_creator
from dqn.agent import Agent
from dqn.replay_buffer import ReplayBuffer
from dqn.train_agent import train_agent
from logger import Logger

config = {
    'device': 'cpu',
    'log_dir': './dqn/results/',
    'log_stdout': True,

    'experiment_name': 'cartpole-1',
    'env_name': 'CartPole-v0',
    'q_network_creator': q_network_mlp_creator,

    # training hyperparameters (from https://github.com/openai/baselines)
    't_max': 100000,                 # maximum training steps(frames)
    'learning_start': 1000,          # number of steps before learning starts
    'replay_capacity': 50000,        # replay buffer size
    'target_update_freq': 500,       # target network update frequency
    'update_freq': 1,                # update frequency between successive SGD
    'gamma': 1.0,                    # discount factor
    'batch_size': 32,                # minibatch size

    # linearly-annealed epsilon-greedy (from https://github.com/openai/baselines)
    'eps_start': 1.0,                # initial epsilon value
    'eps_end': 0.02,                 # final epsilon value
    'eps_end_t': 10000,              # the step(frame) that epsilon reaches final epsilon value

    # Adam optimizer (from https://github.com/openai/baselines)
    'adam_lr': 0.001,                # learning rate
    'adam_eps': 1e-08,               # epsilon

    # evaluation hyperparameters
    'eval_freq': 500,                # evaluation frequency
    'eval_t': 200,                   # number of steps(frames) in evaluation
    'eval_eps': 0.0,                 # value for epsilon-greedy in evaluation
    'eval_complete_episode': True,   # complete episode even if number of evaluation steps exceeds

    'checkpoint_freq': 10000,        # checkpoint for saving model

    # recover training
    'recover': False,
    'recover_t': 0,
    'model_path': './dqn/results/cartpole-1.model',
}

env = gym.make(config['env_name'])
agent = Agent(env, config)
replay_buffer = ReplayBuffer(env, config)
logger = Logger(config)
train_agent(env, agent, replay_buffer, logger, config)
