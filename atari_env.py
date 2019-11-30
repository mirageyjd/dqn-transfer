import gym
import cv2
import numpy as np


def obs_preprocess(unwrapped_obs: np.ndarray) -> np.ndarray:
    # convert into greyscale image
    _obs = cv2.cvtColor(unwrapped_obs, cv2.COLOR_RGB2GRAY)
    # scale into 84 * 84
    _obs = cv2.resize(_obs, (84, 84), interpolation=cv2.INTER_AREA)
    return _obs


# Wrapper of OpenAI gym env class targeted for Atari Game
# Settings are from https://github.com/openai/baselines
class AtariEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, config: dict):
        gym.Wrapper.__init__(self, env)

        self.history_len = config['history_len']
        self.action_repeat = config['action_repeat']
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.history_len, 84, 84), dtype=np.uint8)

        if config['action_mapping_on']:
            self.action_space = gym.spaces.Discrete(len(config['action_mapping']))
            self.action_mapping = np.zeros(len(config['action_mapping']), dtype=np.int)
            action_meaning = self.env.unwrapped.get_action_meanings()
            for i, action_name in enumerate(config['action_mapping']):
                self.action_mapping[i] = action_meaning.index(action_name)
        else:
            self.action_mapping = np.arange(self.action_space.n)

        self.no_op_max = config['no_op_max']

    def reset(self, **kwargs):
        # no-op reset
        unwrapped_obs = self.env.reset(**kwargs)

        num_no_op = np.random.randint(self.history_len - 1, self.no_op_max + 1)
        obs = np.zeros((self.history_len, 84, 84))
        if num_no_op == self.history_len - 1:
            obs[self.history_len - 1] = obs_preprocess(unwrapped_obs)

        for i in range(1, num_no_op + 1):
            unwrapped_obs, reward, done, info = self.env.step(0)
            if done:
                return self.reset(**kwargs)

            index = num_no_op - i
            if index <= self.history_len - 1:
                obs[index] = obs_preprocess(unwrapped_obs)

        return obs

    def step(self, action):
        # Repeat taking same action
        action = self.action_mapping[action]
        obs = np.zeros((self.history_len, 84, 84))
        total_reward = 0.0
        done = False
        info = None
        for i in range(1, self.action_repeat + 1):
            unwrapped_obs, reward, done, info = self.env.step(action)

            index = self.action_repeat - i
            if index <= self.history_len - 1:
                obs[index] = obs_preprocess(unwrapped_obs)
            total_reward += reward

            if done:
                break

        return obs, total_reward, done, info
