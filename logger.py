import os
import torch
import torch.nn as nn


class Logger(object):
    def __init__(self, config):
        self.exp_name = config['experiment_name']
        self.log_name = self.exp_name + '.log'
        self.log_dir = config['log_dir']
        self.eval_t = config['eval_t']

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        log_file = open(self.log_dir + self.log_name, 'w+')
        log_file.write('{}\n'.format(self.exp_name))
        log_file.close()

        self.reward_history = list()
        self.total_reward = 0.0
        self.time_step = 0

    def start(self, time_step: int):
        self.total_reward = 0.0
        self.time_step = time_step

    def record(self, reward: float):
        self.total_reward += reward

    def end(self):
        avg_reward = self.total_reward / self.eval_t
        self.reward_history.append(avg_reward)

        log_file = open(self.log_dir + self.log_name, 'a+')
        log_file.write('[{}, {}]\n'.format(self.time_step, avg_reward))
        log_file.close()

    def save_model(self, model: nn.Module):
        torch.save(model.state_dict(), self.log_dir + self.exp_name + '.model')

    def train_over(self):
        log_file = open(self.log_dir + self.log_name, 'w+')
        log_file.write('{}\n'.format(self.reward_history))
        log_file.close()
