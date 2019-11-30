from typing import Tuple
import torch
import numpy as np
import gym


class ReplayBuffer(object):
    def __init__(self, env: gym.Env, config: dict):
        """
            env: openai gym environment
            n: current size
            capacity: capacity
            last: index of last inserted transition
            s, a, r, s2 : transition data
        """
        self.device = config['device']

        self.n = 0
        self.capacity = config['replay_capacity']
        self.last = -1

        self.s = torch.empty((self.capacity,) + env.observation_space.shape, device=self.device)
        self.a = torch.empty(self.capacity, dtype=torch.long, device=self.device)
        self.r = torch.empty(self.capacity, device=self.device)
        self.s2 = torch.empty((self.capacity,) + env.observation_space.shape, device=self.device)

    # insert a transition tuple (s, a, r, s2)
    def insert(self, trans: Tuple[torch.Tensor, int, float, torch.Tensor]):
        self.last = (self.last + 1) % self.capacity
        if self.n < self.capacity:
            self.n += 1

        self.last = (self.last + 1) % self.capacity
        temp1, self.a[self.last], self.r[self.last], temp2 = trans
        self.s[self.last] = torch.from_numpy(temp1).float().to(self.device)
        self.s2[self.last] = torch.from_numpy(temp2).float().to(self.device)

    # sample a batch
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ind = np.random.randint(0, self.n, size=batch_size)
        return self.s[ind], self.a[ind], self.r[ind], self.s2[ind]
