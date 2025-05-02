import torch.nn as nn
import torch
import torch.nn.functional as F
from RL_Algorithm.RL_base import BaseAlgorithm

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
    def __init__(self, input_dim, hidden_dim, num_actions, device):
        super(Dueling_DQN_network, self).__init__()
        self.device = device
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
        x = x.to(self.device)
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
    
class Dueling_DQN(BaseAlgorithm):
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

        self.previous_weight = torch.zeros_like(self.policy_network.fc1_adv.weight)

    
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
        loss = self.calculate_loss(batch=batch) 
        self.update_policy_network(loss)
        return loss.item()