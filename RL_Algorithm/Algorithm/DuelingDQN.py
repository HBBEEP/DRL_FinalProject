import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class Dueling_DQN_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(Dueling_DQN_network, self).__init__()
        self.num_actions = num_actions

        self.conv1 = ConvBlock(input_dim, hidden_dim)  # input_dim = 16 | hidden_dim = 2048
        self.conv2 = ConvBlock(hidden_dim, hidden_dim) # hidden_dim = 2048 | hidden_dim = 2048
        self.conv3 = ConvBlock(hidden_dim, hidden_dim) # hidden_dim = 2048 | hidden_dim = 2048

        self.fc1_adv = nn.Linear(in_features=hidden_dim*input_dim, out_features= hidden_dim // 4) # input_dim = hidden_dim*input_dim = 2048 * 16 | hidden_dim = hidden_dim // 4 = 2048 // 4
        self.fc1_val = nn.Linear(in_features=hidden_dim*input_dim, out_features= hidden_dim // 4) # input_dim = hidden_dim*input_dim = 2048 * 16 | hidden_dim = hidden_dim // 4 = 2048 // 4

        self.fc2_adv = nn.Linear(in_features=hidden_dim // 4, out_features=num_actions) # in_features = hidden_dim // 4 = 2048 // 4 |  out_features=num_actions = 4
        self.fc2_val = nn.Linear(in_features=hidden_dim // 4, out_features=1) # in_features = hidden_dim // 4 = 2048 // 4 |  out_features=1

        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x