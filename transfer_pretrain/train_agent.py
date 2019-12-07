import numpy as np
import gym
import torch
from agent import Agent
from replay_buffer import ReplayBuffer
from logger import Logger
from tqdm import tqdm
from img_transfer import load_model, img_transfer


def train_agent(env: gym.Env, agent: Agent, replay_buffer: ReplayBuffer, logger: Logger, config: dict,
                config_source: dict, config_target: dict):
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    agent.load_model_from_state_dict_source(torch.load(config_source['model_path']))
    action_mapping = build_action_mapping(config, config_source, config_target)
    # UNIT GAN encoder & decoder
    encoder, decoder, transformer = load_model(config['unit_gan_config'],
                                               config['unit_gan_model'],
                                               config['unit_gan_folder'])

    start_t = 0
    if config['recover']:
        start_t = max(start_t, config['recover_t'] - config['learning_start'])
        config['learning_start'] += start_t
        agent.load_model_from_state_dict_target(torch.load(config['model_path']))
        print('Load target model from ', config['model_path'])
        print('Recover training from step ', config['recover_t'])

    s = env.reset()
    for t in tqdm(range(start_t + 1, config['t_max'] + 1)):
        # sampling from environment
        a = agent.action(s, config['eps'])
        s2, r, done, _ = env.step(a)
        update_target = agent.get_update_target(r, s2, done, config['gamma'])
        # state mapping from source to target
        s2 = img_transfer(encoder, decoder, transformer, s2)
        a = action_mapping[a]
        replay_buffer.insert((s, a, update_target))

        if t > config['learning_start']:
            if t % config['update_freq'] == 0:
                # sample a mini-batch from replay buffer and update q function
                s_batch, a_batch, target_batch = replay_buffer.sample(config['batch_size'])
                agent.train(s_batch, a_batch, target_batch)

            if t % config['checkpoint_freq'] == 0:
                logger.save_model(agent.get_model_target())

        s = env.reset() if done else s2


def build_action_mapping(config: dict, config_source: dict, config_target: dict):
    action_mapping = np.zeros(len(config_source['action_mapping']), dtype=np.int)
    for i, source_action_name in enumerate(config_source['action_mapping']):
        target_action_index = config_target['action_mapping'].index(config['s2t_action_mapping'][source_action_name])
        action_mapping[i] = target_action_index
    return action_mapping
