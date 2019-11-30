from typing import Tuple
import torch
import torch.optim as optim
import numpy as np
import gym


# Q function
class QFunction(object):
    def __init__(self, env: gym.Env, config: dict):
        self.device = config['device']

        self.q_network = config['q_network_creator'](env).to(device=self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['adam_lr'], eps=config['adam_eps'])

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

    # copy parameters from another q function
    def copy(self, q_func):
        self.q_network.load_state_dict(q_func.q_network.state_dict())


# DQN Agent
class Agent(object):
    def __init__(self, env: gym.Env, config: dict):
        self.device = config['device']

        self.env = env
        self.q_func = QFunction(env, config)
        self.target_q_func = QFunction(env, config)
        self.target_q_func.copy(self.q_func)

    # take action under epsilon-greedy policy
    def action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.uniform() <= epsilon:
            return np.random.randint(0, self.env.action_space.n)
        else:
            s_tensor = torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            q_argmax = self.q_func.argmax(s_tensor)
            return q_argmax

    # train the agent with a mini-batch of transition (s, a, r, s2)
    def train(self, s_batch: torch.Tensor, a_batch: torch.Tensor, r_batch: torch.Tensor, s2_batch: torch.Tensor,
              gamma: float):
        # move tensors to training device and set data type of tensors
        s_batch = s_batch.to(device=self.device, dtype=torch.float32)
        a_batch = a_batch.to(device=self.device, dtype=torch.long)
        r_batch = r_batch.to(device=self.device)
        s2_batch = s2_batch.to(device=self.device, dtype=torch.float32)

        target_batch = r_batch + gamma * self.target_q_func.max_batch(s2_batch)
        self.q_func.update(s_batch, a_batch, target_batch)

    # update target q function
    def update_target(self):
        self.target_q_func.copy(self.q_func)
