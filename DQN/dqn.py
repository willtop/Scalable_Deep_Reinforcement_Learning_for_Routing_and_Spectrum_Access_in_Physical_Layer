# This script contains our novel deep reinforcement learning network model implementation
# for the work "Scalable Deep Reinforcement Learning for Routing and Spectrum Access in Physical Layer", 
# available at arxiv.org/abs/2012.11783.

# For any reproduce, further research or development, please kindly cite our paper (TCOM Journal version upcoming soon):
# @misc{rl_routing,
#    author = "W. Cui and W. Yu",
#    title = "Scalable Deep Reinforcement Learning for Routing and Spectrum Access in Physical Layer",
#    month = dec,
#    year = 2020,
#    note = {[Online] Available: https://arxiv.org/abs/2012.11783}
# }


# DQN Model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dueling_DQN_Model(nn.Module):
    def __init__(self, n_input_features, n_actions):
        super().__init__()
        self.n_input_features = n_input_features
        self.fc_1 = nn.Linear(self.n_input_features, 150)
        self.fc_2 = nn.Linear(150, 150)
        self.fc_Q_1 = nn.Linear(150, 100)
        self.fc_Q_2 = nn.Linear(100, 1)
        self.fc_A_1 = nn.Linear(150, 100)
        self.fc_A_2 = nn.Linear(100, n_actions)

    def forward(self, x):
        minibatch_size, n_input_features = x.size()
        assert n_input_features == self.n_input_features
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x_q = F.relu(self.fc_Q_1(x))
        x_q = self.fc_Q_2(x_q)
        x_a = F.relu(self.fc_A_1(x))
        x_a = self.fc_A_2(x_a)
        q_vals = x_q + x_a - torch.mean(x_a, dim=1, keepdim=True)
        return q_vals



