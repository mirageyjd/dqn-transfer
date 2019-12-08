import gym
import cv2
import numpy as np
from PIL import Image


def obs_preprocess(unwrapped_obs: np.ndarray) -> np.ndarray:
    # convert into greyscale image
    _obs = cv2.cvtColor(unwrapped_obs, cv2.COLOR_RGB2GRAY)
    # scale into 84 * 84
    _obs = cv2.resize(_obs, (84, 84), interpolation=cv2.INTER_AREA)
    return _obs


def clipped_reward(reward: float) -> float:
    if reward > 0:
        return 1.0
    elif reward < 0:
        return -1.0
    else:
        return 0.0


# Wrapper class for performing image pre-processing
class TennisPreprocEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        # Mean background pixel values
        self.tennis_mean_img = Image.open('./tennis_mean.png')
        self.tennis_mean_img.load()
        self.tennis_mean_img = np.asarray(self.tennis_mean_img, dtype="float")
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for i in range(4):
            obs[i] = self.tennis_img_preprocess(obs[i])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for i in range(4):
            obs[i] = self.tennis_img_preprocess(obs[i])
        return obs, reward, done, info

    def tennis_img_preprocess(self, img):
        img = img.astype(float)
        # Rotate
        img -= self.tennis_mean_img
        img[img > 255] = 255
        img[img < 0] = 0
        # Crop the score
        for i in range(15):
            img[i,:] = np.zeros((84))
        img = img.astype('uint8')
        return img


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

        return obs, clipped_reward(total_reward), done, info


# Special wrapper for atari tennis: terminate when the first game ends
class AtariTennisWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)

        self.player_score = 0
        self.opponent_score = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.player_score = 0
        self.opponent_score = 0

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if reward != 0:
            if reward > 0:
                self.player_score += 1
            else:
                self.opponent_score += 1

            done = self.game_terminate()

        return obs, reward, done, info

    def game_terminate(self):
        if self.player_score >= 4 and self.player_score - self.opponent_score >= 2:
            return True

        if self.opponent_score >= 4 and self.opponent_score - self.player_score >= 2:
            return True

        return False
