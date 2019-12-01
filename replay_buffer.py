from typing import Tuple
import torch
import numpy as np
import gym

TYPE_MAP = {
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float32): torch.float32,
}


class ReplayBuffer(object):
    def __init__(self, env: gym.Env, config: dict):
        """
            env: openai gym environment
            n: current size
            capacity: capacity
            last: index of last inserted transition
            s, a, r, s2, done: transition data
        """

        self.n = 0
        self.capacity = config['replay_capacity']
        self.last = -1

        self.s = torch.empty((self.capacity,) + env.observation_space.shape,
                             dtype=TYPE_MAP[env.observation_space.dtype])
        self.a = torch.empty(self.capacity, dtype=torch.int64)
        self.r = torch.empty(self.capacity, dtype=torch.float32)
        self.s2 = torch.empty((self.capacity,) + env.observation_space.shape,
                              dtype=TYPE_MAP[env.observation_space.dtype])
        self.done = torch.empty(self.capacity, dtype=torch.uint8)

    # insert a transition tuple (s, a, r, s2, done)
    def insert(self, trans: Tuple[np.ndarray, int, float, np.ndarray, int]):
        self.last = (self.last + 1) % self.capacity
        if self.n < self.capacity:
            self.n += 1
        s_new, a_new, r_new, s2_new, done_new = trans
        self.s[self.last] = torch.from_numpy(s_new)
        self.a[self.last] = a_new
        self.r[self.last] = r_new
        self.s2[self.last] = torch.from_numpy(s2_new)
        self.done[self.last] = done_new

    # sample a mini-batch
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ind = np.random.randint(0, self.n, size=batch_size)
        return self.s[ind], self.a[ind], self.r[ind], self.s2[ind], self.done[ind]
