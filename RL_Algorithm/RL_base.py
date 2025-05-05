import torch
import numpy as np
from Game_2048.board import Board, main_loop
import math,random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class BaseAlgorithm():
    def __init__(
        self,
        device:str,
        policy_network,
        target_network,
        initial_epsilon:float,
        epsilon_decay:float,
        final_epsilon:float,
        learning_rate:float,
        tau:float,
        batch_size:int,
        buffer_size:int,
        soft_update:bool,
        use_scheduler:bool,
    ):
        self.policy_network = policy_network
        self.target_network= target_network
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.soft_update = soft_update
        self.use_scheduler = use_scheduler

        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.98)

        self.memory = ReplayMemory(capacity=self.buffer_size) 

        self.step_done = 0
        self.global_step_done = 0      

    def encode_state(self,current_board:np.ndarray) -> torch.Tensor:
        board_flat = [0 if e == 0 else int(math.log(e, 2)) for e in current_board.flatten()]
        board_flat = torch.LongTensor(board_flat)
        board_flat = torch.nn.functional.one_hot(board_flat, num_classes=16).float().flatten()
        board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2) #.to(device=self.device)
        return board_flat
    
    def select_action(self,encode_state:torch.Tensor,play_mode:bool = False) -> torch.Tensor :
        self.step_done += 1
        self.global_step_done += 1
        if play_mode: 
            sample = 1.0
        else:
            sample = random.random()

        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_network(encode_state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]],device=self.device, dtype=torch.long)
        
    def get_batch_dataset(self):
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        return batch
        
    def update_policy_network(self,loss):
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        if self.use_scheduler:
            if self.step_done >= 200 and self.global_step_done < 1000000:
                self.step_done = 0
                self.scheduler.step()

    def update_target_network(self):
        if self.soft_update:
            policy_net_weight = self.policy_network.state_dict()
            target_net_weight = self.target_network.state_dict()

            for key in policy_net_weight:
                target_net_weight[key] = self.tau * policy_net_weight[key] + (1 - self.tau) * target_net_weight[key]

            self.target_network.load_state_dict(target_net_weight)
            self.policy_network.train()
        else:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.policy_network.train()


    def same_move(self, state, next_state, last_memory):
        return torch.eq(state, last_memory.state).all() and torch.eq(next_state, last_memory.next_state).all()
    
    def epsilon_update(self):
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)


