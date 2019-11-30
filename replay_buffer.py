from typing import Tuple
import torch
import numpy as np
import gym

TYPE_MAP = {
    np.uint8: torch.uint8,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float32: torch.float32,
}


class ReplayBuffer(object):
    def __init__(self, env: gym.Env, config: dict):
        """
            env: openai gym environment
            n: current size
            capacity: capacity
            last: index of last inserted transition
            s, a, r, s2 : transition data
        """

        self.n = 0
        self.capacity = config['replay_capacity']
        self.last = -1

        self.s = torch.empty((self.capacity,) + env.observation_space.shape,
                             dtype=TYPE_MAP[env.observation_space.dtype])
        self.a = torch.empty(self.capacity, dtype=TYPE_MAP[env.action_space.dtype])
        self.r = torch.empty(self.capacity, dtype=torch.float32)
        self.s2 = torch.empty((self.capacity,) + env.observation_space.shape,
                              dtype=TYPE_MAP[env.observation_space.dtype])

    def insert(self, trans: Tuple[np.ndarray, int, float, np.ndarray]):
        """
            insert a transition tuple

            params:
                trans: transition (s, a, r, s2)
        """

        self.last = (self.last + 1) % self.capacity
        if self.n < self.capacity:
            self.n += 1

        self.last = (self.last + 1) % self.capacity
        s_new, a_new, r_new, s2_new = trans
        self.s[self.last] = torch.from_numpy(s_new)
        self.a[self.last] = a_new
        self.r[self.last] = r_new
        self.s2[self.last] = torch.from_numpy(s2_new)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            sample a batch from replay buffer

            params:
                batch_size: the size of mini-batch

            returns:
                a tuple of transition batch (s_batch, a_batch, r_batch. s2_batch)
        """

        ind = np.random.randint(0, self.n, size=batch_size)
        return self.s[ind], self.a[ind], self.r[ind], self.s2[ind]
