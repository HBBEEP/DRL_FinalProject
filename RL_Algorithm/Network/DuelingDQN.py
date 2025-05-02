import torch.nn as nn
import torch
import torch.nn.functional as F

class Dueling_DQN_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, device):

        self.device = device
        self.num_actions = num_actions
        super(Dueling_DQN_network, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim//8, (2,2))  # input_dim = 16 | hidden_dim = 2048 // 8
        self.conv2 = nn.Conv2d(hidden_dim//8, hidden_dim//4, (2,2)) # hidden_dim = 2048 // 8 | hidden_dim = 2048 // 4

        self.fc1_adv = nn.Linear(in_features=hidden_dim, out_features= hidden_dim // 2) # input_dim = hidden_dim = 2048  | hidden_dim = hidden_dim // 2 = 2048 // 2
        self.fc1_val = nn.Linear(in_features=hidden_dim, out_features= hidden_dim // 2) # input_dim = hidden_dim = 2048  | hidden_dim = hidden_dim // 2 = 2048 // 2

        self.fc2_adv = nn.Linear(in_features=hidden_dim // 2, out_features=num_actions) # in_features = hidden_dim // 2 = 2048 // 2 |  out_features=num_actions = 4
        self.fc2_val = nn.Linear(in_features=hidden_dim // 2, out_features=1) # in_features = hidden_dim // 2 = 2048 // 2 |  out_features=1

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Flatten()(x)
        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x