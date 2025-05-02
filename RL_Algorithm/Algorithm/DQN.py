import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from RL_Algorithm.RL_base import BaseAlgorithm

class DQN_network(nn.Module):
    def __init__(self, device:str, hidden_dim:int = 2048):
        self.device = device
        super(DQN_network, self).__init__()
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
            soft_update:bool,
            device:str = 'cuda',
    ) -> None:
        self.device = device
        policy_network = DQN_network(hidden_dim=hidden_dim,device=self.device).to(device=self.device)
        target_network = DQN_network(hidden_dim=hidden_dim,device=self.device).to(device=self.device)
  
        policy_network.train()
        target_network.eval()
        
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
            soft_update = soft_update,
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

        return loss.item()