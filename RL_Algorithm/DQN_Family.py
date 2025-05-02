import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from RL_Algorithm.RL_base import BaseAlgorithm
from RL_Algorithm.Network.DQN import DQN_network
from RL_Algorithm.Network.DoubleDQN import DoubleDQN_network
from RL_Algorithm.Network.DuelingDQN import Dueling_DQN_network



class DQNFamily(BaseAlgorithm):
    def __init__(
            self,
            algorithm:str,
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

        if algorithm == "DQN":
            policy_network = DQN_network(hidden_dim=hidden_dim,device=self.device).to(device=self.device)
            target_network = DQN_network(hidden_dim=hidden_dim,device=self.device).to(device=self.device)
        elif algorithm == "DoubleDQN":
            policy_network = DoubleDQN_network(hidden_dim=hidden_dim,device=self.device).to(device=self.device)
            target_network = DoubleDQN_network(hidden_dim=hidden_dim,device=self.device).to(device=self.device)
        elif algorithm == "DuelingDQN":
            policy_network = Dueling_DQN_network(input_dim=16, hidden_dim=hidden_dim, num_actions=4, device=self.device).to(device=self.device)
            target_network = Dueling_DQN_network(input_dim=16, hidden_dim=hidden_dim, num_actions=4, device=self.device).to(device=self.device)

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