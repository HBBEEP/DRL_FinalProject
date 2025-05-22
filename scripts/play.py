import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm,torch
from Game_2048.board import Board, main_loop
from RL_Algorithm.RL_base import BaseAlgorithm
from RL_Algorithm.DQN_Family import DQNFamily
from utils.board_visualizer import Board_Animator

import yaml,json
import argparse
import pandas as pd

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
parser.add_argument('--play', type=str, required=True, help='Play name (e.g., play_1)')
parser.add_argument('--debug', type=str2bool, default=False, help='True if you want to track the training process')
args = parser.parse_args()

debug_flag = args.debug

# Load the YAML file
with open(f'params/{args.algo}.yaml', 'r') as file:
    config = yaml.safe_load(file)


selected_config = config[args.algo][args.play]
print(selected_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

board_env = Board()
board_visualizer = Board_Animator()

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


#//////////////// load weight ///////////////////////////
agent.policy_network.load_state_dict(torch.load(selected_config["policy_network_weight"]))
agent.policy_network.eval()
agent.target_network.load_state_dict(torch.load(selected_config["target_network_weight"]))
agent.target_network.eval()
#////////////////////////////////////////////////////////

total_scores = []
best_tile_list = []
all_episode_action = []
average = 0

for episode in range(selected_config['n_episodes']):
    board_env.reset()
    done = False
    cumulative_reward = 0
    cumulative_loss = 0
    update_count = 0
    step_count = 0
    duplicate = False
    action_list = []

    state = agent.encode_state(board_env.board)
    non_valid_count, valid_count = 0, 0
    while not done:
        # agent stepping
        action = agent.select_action(state,play_mode=True)
        action_list.append(action.item())
        old_score = board_env.total_score

        # env stepping
        if non_valid_count >= 50:
            action = torch.randint(low=0, high=5, size=(1,))
            non_valid_count = 0
            
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
            board_visualizer.update(board_env.board,score=board_env.total_score,delay=0.02)

        if done:
            board_visualizer.done()
            print(f"=============== Episode : {episode} ======================")
            print(f"Epsilon : {agent.epsilon}")
            print(f"Episode score: {board_env.total_score}")
            print(f"Non valid move count: {non_valid_count}")
            print(f"Valid move count: {valid_count}")
            print(f"Cumulative reward: {cumulative_reward}")
            print(f"Training device: {agent.device}")
            total_scores.append(board_env.total_score)
            best_tile_list.append(board_env.board.max())
            all_episode_action.append(action_list)
            if episode > 50:
                average = sum(total_scores[-50:]) / 50
                print(f"50 episode running average: {average}")

                data = {
                    'episode_score': total_scores,
                    'best_tile_score': best_tile_list,
                }

                df = pd.DataFrame(data)
                df.to_csv(os.path.join(selected_config["save_path"],'play_log.csv'), index=False)  # index=False to not save row numbers

                action_data = {str(idx): value for idx, value in enumerate(all_episode_action)}
                with open(os.path.join(selected_config["save_path"],'play_action_log.json'), 'w') as f:
                    json.dump(action_data, f, indent=4)

            print("==================================================")
            print("\n")
            break


data = {
    'episode_score': total_scores,
    'best_tile_score': best_tile_list,
}
df = pd.DataFrame(data)
df.to_csv(os.path.join(selected_config["save_path"],'play_log.csv'), index=False)  # index=False to not save row numbers

action_data = {str(idx): value for idx, value in enumerate(all_episode_action)}
with open(os.path.join(selected_config["save_path"],'play_action_log.json'), 'w') as f:
    json.dump(action_data, f, indent=4)


        