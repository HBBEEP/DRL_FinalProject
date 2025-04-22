import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm,torch
from Game_2048.board import Board, main_loop
from RL_Algorithm.RL_base import BaseAlgorithm
from RL_Algorithm.Algorithm.DQN import DQN
import yaml
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Select algorithm and experiment.')
parser.add_argument('--algo', type=str, required=True, help='Name of the algorithm (e.g., DuelingDQN)')
parser.add_argument('--exp', type=str, required=True, help='Experiment name (e.g., experiment_1)')
parser.add_argument('--debug', type=bool, default=False, help='True if you want to track the training process')
args = parser.parse_args()

debug_flag = args.debug

# Load the YAML file
with open(f'params/{args.algo}.yaml', 'r') as file:
    config = yaml.safe_load(file)


selected_config = config[args.algo][args.exp]
print(selected_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

board_env = Board()

#//////////////// Algorithm selection ///////////////////////////
agent = DQN(initial_epsilon=selected_config['initial_epsilon'],
            epsilon_decay=selected_config['epsilon_decay'],
            final_epsilon=selected_config['final_epsilon'],
            learning_rate=selected_config['learning_rate'],
            discount_factor=selected_config['discount_factor'],
            tau=selected_config['tau'],
            batch_size=selected_config['batch_size'],
            buffer_size=selected_config['buffer_size'],
            device=device)
#///////////////////////////////////////////////////////////////

total_scores = []
best_tile_list = []

for episode in range(selected_config['n_episodes']):
    board_env.reset()
    done = False
    cumulative_reward = 0
    step_count = 0

    state = agent.encode_state(board_env.board)
    non_valid_count, valid_count = 0, 0
    while not done:
        # agent stepping
        action = agent.select_action(state)
        old_score = board_env.total_score

        # env stepping
        board_env.step(direction=action.item())
        done = board_env.is_game_over()
        
        reward = (board_env.total_score - old_score)
        reward = torch.tensor([reward], device=agent.device)                    # --> reward terms design ? 
        cumulative_reward += reward

        # Observe new state
        if not done:
            next_state = agent.encode_state(board_env.board)
        else:
            next_state = None
        
        if next_state != None and torch.eq(state, next_state).all():
            non_valid_count += 1
            reward -= 10
        else:
            valid_count += 1
        
        if next_state == None or len(agent.memory) == 0 or not agent.same_move(state, next_state, agent.memory.memory[-1]):
            action = action.to(dtype=torch.long)
            agent.memory.push(state, action , next_state, reward)
        
        # if len(agent.memory) >= agent.batch_size:
        #     loss = agent.update()       

        # if step_count%target_update_interval == 0:
        #     agent.update_target_network()

        step_count += 1
        state = next_state

        if done:
            for _ in range(100):
                loss = agent.update()

            print(board_env.board)
            print(f"Episode Score: {board_env.total_score}")
            print(f"Non valid move count: {non_valid_count}")
            print(f"Valid move count: {valid_count}")
            print(f"policy network loss: {loss}")
            total_scores.append(board_env.total_score)
            best_tile_list.append(board_env.board.max())
            if episode > 50:
                average = sum(total_scores[-50:]) / 50
                print(f"50 episode running average: {average}")
            break

    if episode%selected_config['target_update_interval'] == 0:
        agent.update_target_network()
        