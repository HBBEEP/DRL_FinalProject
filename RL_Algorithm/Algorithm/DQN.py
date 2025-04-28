import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from RL_Algorithm.RL_base import BaseAlgorithm

# Define the neural network
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

class DQN_network(nn.Module):
    def __init__(self,hidden_dim:int = 512):
        super(DQN_network, self).__init__()
        self.conv1 = ConvBlock(16, hidden_dim)
        self.conv2 = ConvBlock(hidden_dim, hidden_dim)
        self.conv3 = ConvBlock(hidden_dim, hidden_dim)
        self.dense1 = nn.Linear(hidden_dim * 16, int(hidden_dim/2))
        self.dense6 = nn.Linear(int(hidden_dim/2), 4)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense6(x)

class DQN(BaseAlgorithm):
    def __init__(
            self,
            initial_epsilon:float,
            epsilon_decay:float,
            final_epsilon:float,
            learning_rate:float,
            discount_factor:float,
            tau:float,
            batch_size:int,
            buffer_size:int,
            hidden_dim:int,
            device:str = 'cuda',
    ) -> None:

        policy_network = DQN_network(hidden_dim=hidden_dim).to(device=device)
        target_network = DQN_network(hidden_dim=hidden_dim).to(device=device)
        
        self.device = device
        self.discount_factor = discount_factor

        super().__init__(
            policy_network = policy_network,
            target_network= target_network,
            initial_epsilon = initial_epsilon,
            epsilon_decay = epsilon_decay,
            final_epsilon = final_epsilon,
            learning_rate = learning_rate,
            tau = tau,
            batch_size = batch_size,
            buffer_size = buffer_size,
            device=device
        )

        self.previous_weight = torch.zeros_like(self.policy_network.dense1.weight)
    
    def calculate_loss(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # print(state_batch.shape)
        # print(action_batch.shape)
        # print(reward_batch.shape)

        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss
        
    def update(
        self,
    ):
        """
        Calculate loss from each algorithm
        """
        batch = self.get_batch_dataset()
        if batch == None:
            return None
        loss = self.calculate_loss(batch=batch) 
        self.update_policy_network(loss)

        # print("Weights changed:", not torch.allclose(self.previous_weight, self.policy_network.dense1.weight, rtol=1e-05, atol=1e-08))
        # self.previous_weight = self.policy_network.dense1.weight.detach().clone()

        return loss.item()