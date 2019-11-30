import torch.nn as nn
import gym
import torch


class QNetworkAtari(nn.Module):
    def __init__(self, env: gym.Env):
        super(QNetworkAtari, self).__init__()
        # CNN: 4 * 84 * 84 -> 32 * 20 * 20 -> 64 * 9 * 9 -> 64 * 7 * 7
        self.conv_1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(True),
        )

        # Fully connected: 64 * 7 * 7 = 3136 -> 512 -> num_action
        self.fc_1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(True)
        )
        self.fc_2 = nn.Linear(512, env.action_space.n)

    def forward(self, state_in):
        q_out = self.conv_1(state_in)
        q_out = self.conv_2(q_out)
        q_out = self.conv_3(q_out)
        q_out = torch.flatten(q_out, start_dim=1)
        q_out = self.fc_1(q_out)
        q_out = self.fc_2(q_out)
        return q_out


class QNetworkMlp(nn.Module):
    def __init__(self, env: gym.Env):
        super(QNetworkMlp, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(True)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True)
        )
        self.fc_3 = nn.Linear(64, env.action_space.n)

    def forward(self, state_in):
        q_out = self.fc_1(state_in)
        q_out = self.fc_2(q_out)
        q_out = self.fc_3(q_out)
        return q_out


def q_network_atari_creator(env: gym.Env):
    return QNetworkAtari(env)


def q_network_mlp_creator(env: gym.Env):
    return QNetworkMlp(env)
