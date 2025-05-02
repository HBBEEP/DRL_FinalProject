import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from RL_Algorithm.RL_base import BaseAlgorithm

class DoubleDQN_network(nn.Module):
    def __init__(self, device:str, hidden_dim:int = 2048):
        self.device = device
        super(DoubleDQN_network, self).__init__()
        # self.conv1 = nn.Conv2d(16, hidden_dim//8, (2,2))
        # self.conv2 = nn.Conv2d(hidden_dim//8, hidden_dim//4, (2,2))
        # self.dense1 = nn.Linear(hidden_dim, hidden_dim//2)
        # self.dense6 = nn.Linear(hidden_dim//2, hidden_dim//4)
        # self.dense7 = nn.Linear(hidden_dim//4, 4)
        pass

    def forward(self, x):
        # x = x.to(self.device)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = x.reshape(x.size(0), -1)
        # x = F.relu(self.dense1(x))
        # x = F.relu(self.dense6(x))
        # x = self.dense7(x)
        pass
        # return x