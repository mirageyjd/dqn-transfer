import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


# Q function
class QFunction(object):
    def __init__(self, env: gym.Env, config: dict):
        self.device = config['device']

        self.q_network = config['q_network_creator'](env).to(device=self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['adam_lr'], eps=config['adam_eps'])

    # return max(q(s,a))
    def max(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q_s = self.q_network(state)
        q_max, q_argmax = q_s.max(1)
        return q_max.item()

    # return argmax(q(s,a))
    def argmax(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q_s = self.q_network(state)
        q_max, q_argmax = q_s.max(1)
        return q_argmax.item()

    def max_batch(self, s_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_s = self.q_network(s_batch)
        q_max, q_argmax = q_s.max(1)
        return q_max

    # update network
    def update(self, s_batch: torch.Tensor, a_batch: torch.Tensor, target_batch: torch.Tensor):
        q_s = self.q_network(s_batch)
        est_batch = torch.gather(q_s, 1, torch.unsqueeze(a_batch, 1)).squeeze()
        loss = ((est_batch - target_batch) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # load a network model
    def load_model(self, q_network: nn.Module):
        self.q_network.load_state_dict(q_network.state_dict())

    # load a network model from state dict
    def load_model_from_state_dict(self, state_dict: dict):
        self.q_network.load_state_dict(state_dict)

    # retrieve the network model
    def get_model(self):
        return self.q_network


# DQN Agent
class Agent(object):
    def __init__(self, env_source: gym.Env, env_target: gym.Env, config: dict):
        self.device = config['device']

        self.env_source = env_source
        self.env_target = env_target

        self.q_func_source = QFunction(env_source, config)
        self.q_func_target = QFunction(env_target, config)

    # take action under epsilon-greedy policy
    def action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.uniform() <= epsilon:
            return np.random.randint(0, self.env_source.action_space.n)
        else:
            s_tensor = torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            q_argmax = self.q_func_source.argmax(s_tensor)
            return q_argmax

    # get update target y_j = r + gamma * max(q(s2, a2)) for transition tuple (s, a, r, s2)
    def get_update_target(self, r: float, s2: np.ndarray, done: int, gamma: float):
        s2_tensor = torch.from_numpy(s2).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        q_max = self.q_func_source.max(s2_tensor)
        y_j = r + gamma * q_max * (1 - done)
        return y_j

    # train the agent with a mini-batch of transition (s, a, r, s2, done)
    def train(self, s_batch: torch.Tensor, a_batch: torch.Tensor, target_batch: torch.Tensor):
        # move tensors to training device and set data type of tensors
        s_batch = s_batch.to(device=self.device, dtype=torch.float32)
        a_batch = a_batch.to(device=self.device)
        target_batch = target_batch.to(device=self.device)

        self.q_func_target.update(s_batch, a_batch, target_batch)

    # load a q model for target environment
    def load_model_target(self, q_network: nn.Module):
        self.q_func_target.load_model(q_network)

    # load a q model from state dict for target environment
    def load_model_from_state_dict_target(self, state_dict: dict):
        self.q_func_target.load_model_from_state_dict(state_dict)

    # load a q model for source environment
    def load_model_source(self, q_network: nn.Module):
        self.q_func_source.load_model(q_network)

    # load a q model from state dict for source environment
    def load_model_from_state_dict_source(self, state_dict: dict):
        self.q_func_source.load_model_from_state_dict(state_dict)

    # retrieve q model for target environment
    def get_model_target(self):
        return self.q_func_target.get_model()
