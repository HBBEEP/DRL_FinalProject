import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from RL_Algorithm.RL_base import BaseAlgorithm

class DQN_network_mini(nn.Module):
    def __init__(self, device:str, hidden_dim:int = 2048):
        self.device = device
        super(DQN_network_mini, self).__init__()
        self.conv1 = nn.Conv2d(16, hidden_dim//8, (2,2))
        self.conv2 = nn.Conv2d(hidden_dim//8, hidden_dim//4, (2,2))
        self.dense1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.dense6 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.dense7 = nn.Linear(hidden_dim//4, 4)
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense6(x))
        x = self.dense7(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(ConvBlock, self).__init__()
        self.device = device
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        x = x.to(self.device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN_network(nn.Module):

    def __init__(self, hidden_dim, device):
        super(DQN_network, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.conv1 = ConvBlock(16, self.hidden_dim, device=self.device)
        self.conv2 = ConvBlock(self.hidden_dim, self.hidden_dim, device=self.device)
        self.conv3 = ConvBlock(self.hidden_dim, self.hidden_dim, device=self.device)
        self.dense1 = nn.Linear(self.hidden_dim * 16, self.hidden_dim//2)
        self.dense2 = nn.Linear(self.hidden_dim//2, 4)
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense2(x)