# Wrapper of OpenAI gym env class targeted for Atari Game
import numpy as np
import os
from collections import deque
import gym
from gym import spaces
import cv2
import numpy as np
from skimage import transform

def scale_lumininance(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])


def state_preprocess(s):
    # convert into greyscale image
    _state = scale_lumininance(s)
    # Scale into 84*84
    _state = transform.resize(_state, (84, 84))
    _state = np.moveaxis(_state, 1, 0)
    return _state[np.newaxis, :]


class AtariEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

    def reset(self, **kwargs):
        res = np.zeros((4, 84, 84))
        s = self.env.reset()
        for i in range(3):
            res[i] = state_preprocess(s)
        return res

    def step(self, ac):
        # Proceed 4 steps
        res = np.zeros((4, 84, 84))
        total_r = 0
        done = False
        info = None
        for i in range(3):
            s, r, done, info = self.env.step(ac)
            res[i] = state_preprocess(s)
            total_r += r
            if done:
                break
        return res, total_r, done, info
