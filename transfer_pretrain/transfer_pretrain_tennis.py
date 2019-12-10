import gym
from transfer_pretrain.agent import Agent
from transfer_pretrain.replay_buffer import ReplayBuffer
from transfer_pretrain.train_agent import train_agent
from transfer_pretrain.atari_env import AtariEnv, AtariTennisWrapper
from logger import Logger
from network import q_network_atari_creator

config_source = {
    'env_name': 'PongNoFrameskip-v4',

    # action mapping: restrict action space
    'action_mapping_on': True,
    'action_mapping': ['NOOP', 'RIGHT', 'LEFT'],

    # deepmind-style atari (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    'frame_skip': 4,  # frame skip
    'action_repeat': 4,  # number of actions repeated after the agent takes each action
    'history_len': 4,  # agent history length
    'no_op_max': 30,  # maximum number of no-op at the start of each episode

    # pretrain model
    'model_path': './transfer_pretrain/source_model/pong.model'
}

config_target = {
    'env_name': 'TennisNoFrameskip-v4',

    # action mapping: restrict action space
    'action_mapping_on': True,
    'action_mapping': ['FIRE', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE'],

    # deepmind-style atari (from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    'frame_skip': 4,  # frame skip
    'action_repeat': 4,  # number of actions repeated after the agent takes each action
    'history_len': 4,  # agent history length
    'no_op_max': 30,  # maximum number of no-op at the start of each episode
}

config = {
    'device': 'cuda',
    'log_dir': './transfer_pretrain/target_model/',
    'log_stdout': False,

    'experiment_name': 'tennis-pretrain',
    'q_network_creator': q_network_atari_creator,

    # transfer parameters
    's2t_action_mapping': {
        'NOOP': 'FIRE',
        'RIGHT': 'LEFTFIRE',
        'LEFT': 'RIGHTFIRE',
    },

    # Pretrained UNIT GAN model
    'unit_gan_config': './transfer_pretrain/UNIT/configs/unit_atari_folder.yaml',
    'unit_gan_model': './transfer_pretrain/unit_gan_model/gen_00450000.pt',
    'unit_gan_folder': './transfer_pretrain/UNIT',

    # pretrain hyperparameters
    't_max': 5000000,                # maximum training steps(frames)
    'learning_start': 50000,         # number of steps before learning starts
    'replay_capacity': 1000000,      # replay buffer size
    'update_freq': 4,                # update frequency between successive SGD
    'gamma': 0.99,                   # discount factor
    'batch_size': 32,                # minibatch size
    'eps': 0.05,                     # epsilon-greedy

    # Adam optimizer (from https://arxiv.org/pdf/1710.02298.pdf)
    'adam_lr': 0.0000625,            # learning rate
    'adam_eps': 0.00015,             # epsilon

    'checkpoint_freq': 1000000,      # checkpoint for saving model

    # recover training
    'recover': False,
    'recover_t': 0,
    'recover_model_path': './transfer_pretrain/target_model/tennis-pretrain.model',
}

env_source = AtariEnv(gym.make(config_source['env_name']), config_source)
env_target = AtariTennisWrapper(gym.make(config_target['env_name']))
env_target = AtariEnv(env_target, config_target)
agent = Agent(env_source, env_target, config)
replay_buffer = ReplayBuffer(env_target, config)
logger = Logger(config)
train_agent(env_source, agent, replay_buffer, logger, config, config_source, config_target)
