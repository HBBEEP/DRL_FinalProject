import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import math

from collections import namedtuple, deque
from itertools import count

import os
import sys


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Game_2048.board import Board

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the replay buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def encode_state(board):
  board_flat = [0 if e == 0 else int(math.log(e, 2)) for e in board.flatten()]
  board_flat = torch.LongTensor(board_flat)
  board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()
  board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
  return board_flat

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        x = x.to(device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(16, 1024)
        self.conv2 = ConvBlock(1024, 1024)
        self.conv3 = ConvBlock(1024, 1024)
        self.dense1 = nn.Linear(1024 * 16, 512)
        self.dense6 = nn.Linear(512, 4)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense6(x)
    

# Neural Network Initialisation and utilities
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 0.9999
TARGET_UPDATE = 20
LEARNING_RATE = 5e-5
NUM_EPISODES = 20000
BUFFER_SIZE = 50000

n_actions = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

# torch.save(policy_net.state_dict(), '/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/policy_net.pth')
# torch.save(target_net.state_dict(), '/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/target_net.pth')

# policy_net.load_state_dict(torch.load('/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/policy_net.pth'))
# target_net.load_state_dict(torch.load('/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/target_net.pth'))
target_net.eval()
policy_net.train()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
memory = ReplayMemory(BUFFER_SIZE)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START * (EPS_DECAY ** steps_done))
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # print(state_batch.shape)
    # print(action_batch.shape)
    # print(reward_batch.shape)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(f"loss : {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # global steps_done
    # if steps_done % 5000 == 0 and steps_done < 1000000:
    #   print("Learning rate changed.")
    #   scheduler.step()
    return loss

def same_move(state, next_state, last_memory):
  return torch.eq(state, last_memory.state).all() and torch.eq(next_state, last_memory.next_state).all()

def main():
    game = Board()
    total_scores, best_tile_list = [], []
    

    for i_episode in range(NUM_EPISODES):
        print(f"Episode {i_episode}")
        game.reset()
        print(game.board)
        state = encode_state(game.board).float()
        duplicate = False
        non_valid_count, valid_count = 0, 0
        cumulative_reward = 0
        for t in count():
            # Select and perform an action
            action = select_action(state)
            old_score = game.total_score
            old_max = game.board.max()
            game.step(action.item())

            done = game.is_game_over()

            reward = (game.total_score - old_score)
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = encode_state(game.board).float()
            else:
                next_state = None
            
            if next_state != None and torch.eq(state, next_state).all():
                non_valid_count += 1
                reward -= 10
            else:
                valid_count += 1

            # Store the transition in memory
            if next_state != None and duplicate and not torch.eq(state, next_state).all():
              duplicate = False


            if not duplicate:
                if next_state == None or len(memory) == 0 or not same_move(state, next_state, memory.memory[-1]):
                    # print(state)
                    # print(state.shape)
                    cumulative_reward += reward.item()
                    memory.push(state, action, next_state, reward)
            
            if next_state != None:
              duplicate = torch.eq(state, next_state).all()
            
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # optimize_model()
            
            if done:
                cumulative_loss = 0
                update_count = 0
                for _ in range(100):
                    loss = optimize_model()
                    cumulative_loss += loss.item()
                    update_count += 1

                print(game.board)
                print(f"Episode Score: {game.total_score}")
                print(f"Non valid move count: {non_valid_count}")
                print(f"Valid move count: {valid_count}")
                print(f"average_loss: {cumulative_loss/update_count}")
                print(f"cumurative_reward : {cumulative_reward}")
                dummy = encode_state(game.board)
                # print(dummy)
                # print(dummy.shape)
                total_scores.append(game.total_score)
                best_tile_list.append(game.board.max())
                if i_episode > 50:
                    average = sum(total_scores[-50:]) / 50
                    print(f"50 episode running average: {average}")
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            policy_net.train()
        
        # if i_episode % 100 == 0:
            # torch.save(policy_net.state_dict(), '/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/policy_net.pth')
            # torch.save(target_net.state_dict(), '/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/target_net.pth')

    print('Complete')
    # torch.save(policy_net.state_dict(), '/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/policy_net.pth')
    # torch.save(target_net.state_dict(), '/home/hbbeep-p/Documents/GitHub/DRL_FinalProject/weight/DQN/experiment_1/target_net.pth')

    print(total_scores)
    print(best_tile_list)

if __name__ == "__main__":
    main()