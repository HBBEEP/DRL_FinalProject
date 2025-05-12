import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm,torch
from Game_2048.board import Board, main_loop
from RL_Algorithm.RL_base import BaseAlgorithm
from RL_Algorithm.DQN_Family import DQNFamily
from utils.reward_func import full_score_reward, merge_count_reward, guide_score_reward

from utils.board_visualizer import Board_Animator
import yaml,json
import argparse
import pandas as pd
import time

def str2bool(value):
    """Convert a string to a boolean."""
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Select algorithm and experiment.')
parser.add_argument('--algo', type=str, required=True, help='Name of the algorithm (e.g., DuelingDQN)')
parser.add_argument('--exp', type=str, required=True, help='Experiment name (e.g., experiment_1)')
parser.add_argument('--debug', type=str2bool, default=False, help='True if you want to track the training process')
args = parser.parse_args()

debug_flag = args.debug
print(args.debug)
print(type(args.debug))

# Load the YAML file
with open(f'params/{args.algo}.yaml', 'r') as file:
    config = yaml.safe_load(file)


selected_config = config[args.algo][args.exp]
print(selected_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

board_env = Board()
board_visualizer = Board_Animator()

use_preset_board = selected_config['use_preset_board']
max_preset_tile = selected_config['max_preset_tile']
if (use_preset_board):
    max_preset_tile = selected_config['max_preset_tile']
    board_env.set_max_tile(max_preset_tile)

if not os.path.exists(selected_config["save_path"]):
    os.makedirs(selected_config["save_path"])

#//////////////// Algorithm selection ///////////////////////////
agent = DQNFamily(algorithm=args.algo,
                initial_epsilon=selected_config['initial_epsilon'],
                epsilon_decay=selected_config['epsilon_decay'],
                final_epsilon=selected_config['final_epsilon'],
                learning_rate=selected_config['learning_rate'],
                discount_factor=selected_config['discount_factor'],
                tau=selected_config['tau'],
                batch_size=selected_config['batch_size'],
                buffer_size=selected_config['buffer_size'],
                hidden_dim=selected_config['hidden_dim'],
                soft_update=selected_config['soft_update'],
                use_scheduler=selected_config['use_scheduler'],
                device=device)
#///////////////////////////////////////////////////////////////

total_scores = []
best_tile_list = []
loss_list = []
all_episode_action = []
best_scores = 0
average = 0

start_time = time.time()

for episode in range(selected_config['n_episodes']):
    board_env.reset()

    if use_preset_board:
        board_env.preset_board() 
        
    done = False
    cumulative_reward = 0
    cumulative_loss = 0
    update_count = 0
    step_count = 0
    duplicate = False
    action_list = []

    state = agent.encode_state(board_env.board).float()
    non_valid_count, valid_count = 0, 0
    while not done:
        # agent stepping
        action = agent.select_action(state)
        action_list.append(action.item())
        old_score = board_env.total_score
        old_board = board_env.board

        # env stepping
        board_env.step(direction=action.item())
        new_board = board_env.board
        done = board_env.is_game_over()
        
        if selected_config["reward_func"] == "Full_score":
            reward = full_score_reward(board_total_score=board_env.total_score,
                                       old_score=old_score,
                                       device=device)
        elif selected_config["reward_func"] == "Merge_score":
            reward = merge_count_reward(board_total_score=board_env.total_score,
                                        old_score=old_score,
                                        tile_merge=board_env.tile_merge,
                                        device=device)
        elif selected_config["reward_func"] == "Guide_score":
            reward = guide_score_reward(board_total_score=board_env.total_score,
                                        old_score=old_score,
                                        old_board=old_board,
                                        new_board=new_board,
                                        device=device)
        
        cumulative_reward += reward.item()

        # Observe new state
        if not done:
            next_state = agent.encode_state(board_env.board).float()
        else:
            next_state = None
        
        if next_state != None and torch.eq(state, next_state).all():
            non_valid_count += 1
            if selected_config["reward_func"] == "Full_score" or selected_config["reward_func"] == "Guide_score":
                reward -= 10
            elif selected_config["reward_func"] == "Merge_score":
                reward -= 0.01
        else:
            valid_count += 1
        
        # Store the transition in memory
        if next_state != None and duplicate and not torch.eq(state, next_state).all():
            duplicate = False

        if not duplicate:
            if next_state == None or len(agent.memory) == 0 or not agent.same_move(state, next_state, agent.memory.memory[-1]):
                cumulative_reward += reward.item()
                agent.memory.push(state, action, next_state, reward)
        
        if next_state != None:
            duplicate = torch.eq(state, next_state).all()

        step_count += 1
        state = next_state

        if debug_flag:
            board_visualizer.update(board_env.board,board_env.total_score,delay=0.0)

        if done:
            cumulative_loss = 0
            update_count = 0
            for _ in range(100):
                loss = agent.update()
                update_count += 1
                if loss != None:
                    cumulative_loss += loss

            board_visualizer.done()
            print(f"=============== Episode : {episode} ======================")
            print(f"Epsilon : {agent.epsilon}")
            print(f"Learning rate : {agent.policy_optimizer.param_groups[0]['lr']}")
            print(f"Episode score: {board_env.total_score}")
            print(f"Non valid move count: {non_valid_count}")
            print(f"Valid move count: {valid_count}")
            print(f"Average loss: {cumulative_loss/update_count}")
            print(f"Cumulative reward: {cumulative_reward}")
            print(f"Training device: {agent.device}")
            total_scores.append(board_env.total_score)
            best_tile_list.append(board_env.board.max())
            loss_list.append(loss)
            all_episode_action.append(action_list)
            if episode > 50:
                average = sum(total_scores[-50:]) / 50
                print(f"50 episode running average: {average}")

                data = {
                    'episode_score': total_scores,
                    'best_tile_score': best_tile_list,
                    'loss' : loss_list
                }

                df = pd.DataFrame(data)
                df.to_csv(os.path.join(selected_config["save_path"],'train_log.csv'), index=False)  # index=False to not save row numbers

                action_data = {str(idx): value for idx, value in enumerate(all_episode_action)}
                with open(os.path.join(selected_config["save_path"],'train_action_log.json'), 'w') as f:
                    json.dump(action_data, f, indent=4)

            print("==================================================")
            print("\n")

            agent.epsilon_update()

    # Update the target network, copying all weights and biases in DQN
    if episode % selected_config["target_update_interval"] == 0:
        agent.update_target_network()

    if average > best_scores:
        best_scores = average
        torch.save(agent.policy_network.state_dict(), os.path.join(selected_config["save_path"],'policy_best.pth'))
        torch.save(agent.target_network.state_dict(), os.path.join(selected_config["save_path"],'target_best.pth'))


print(f"Training complete in {time.time()-start_time}")

torch.save(agent.policy_network.state_dict(), os.path.join(selected_config["save_path"],'policy_last.pth'))
torch.save(agent.target_network.state_dict(), os.path.join(selected_config["save_path"],'target_last.pth'))




    
        