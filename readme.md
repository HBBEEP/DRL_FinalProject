# Comparative DQN Approaches for 2048 Puzzle Game
FRA 503: Deep Reinforcement Learning for Robotics

## Member
1. Kullakant Kaewkallaya 64340500006
2. Peerawat Santifuengkul 64340500043
3. Monsicha Sopitlaptana 64340500071


## Introduction

2048 is a sliding tile puzzle game. The objective of the game is to slide numbered tiles on a grid to combine them and create a tile with the number 2048. This game is played on a plain 4×4 grid, with numbered tiles that slide when a player moves them using the four arrow keys. 

In this project, we will implement Deep Reinforcement Learning to play the 2048 game using different algorithms: Deep Q-Network, Double Deep Q-Network, and Dueling Deep Q-Network to compare the performance of each algorithm, including differences in reward functions and experimental setups.

## Challenges

A key challenge in training a 2048-playing model is that the states it encounters depend on its current performance. An untrained or weak model often fails early, rarely reaching higher-value tiles like 512 or 1024. As a result, it mostly sees early-game states and struggles to learn strategies for the late game. This slows down training, as the model needs more time and experience to handle advanced scenarios it rarely encounters.

## Technique

### Literature review

description description description

### Game Environment 

#### State
The game is played on a 4×4 grid filled with numbered tiles in the form of 2^x. Therefore, the number of possible states can be modeled as 16^16, since there are 16 grid spaces and each can be filled with a value starting from 0, 2^1, 2^2, ..., up to 2^15. The actual number of states may be even larger depending on how the state space is defined, such as treating empty tiles differently or allowing values beyond 2^15.

To encode the state of the board before feeding it into the neural network, we first flatten the board and replace each non-zero value with its base-2 logarithm, while keeping zeros as 0. Next, we apply one-hot encoding with 16 classes, then flatten the resulting tensor. Finally, we reshape the board to (1, 4, 4, 16) and permute the dimensions to (1, 16, 4, 4) to match the expected input format for the neural network.

#### Action

The player has four possible actions to move the tiles: left, up, right, or down. These actions can be categorized into two types: valid moves, which can be executed in the current state of the board, and non valid moves, which cannot be executed given the current board configuration.

#### Termination

The game terminates when there are no empty spots left on the board and no further merges are possible.

### Algorithms

#### 1. Vanilla DQN 

description description description

#### 2. Double DQN

description description description

#### 3. Dueling DQN 

description description description


## Experiment

#### Reward Functions

We constructed experiments on different reward functions, which were referenced from the literature review and our hypotheses. The reward functions used in this project include:

- **full_score_reward**
  
From the literature review about "xxxxxx", we use different board scores from the stepping process as reward terms. This represents the impact of tile merges, where merging higher tiles returns more reward to the environment. However, this might cause the network to skip the global minima. The penalty reward term was defined by 'non-valid moves' that subtract 10 from the reward each time a non-valid move occurs.

```math
reward = boardscore_{new} - boardscore_{old}
```

- **merge_count_reward**
  
From another literature review about 'xxxxxx' that applied normalization to the reward term by using the ratio of the number of tiles merged in that step. Building on this idea, we also added a score board term that was calculated as a ratio with the goal score. This represents not only how many tiles were merged in that step but also provides the impact of the tile scores that were merged at that step. The penalty reward term was defined as same as the full_score_reward but reduce subtract value from 10 to 0.01 follow the reward size.

```math
reward = \frac{n_{tilesmerged}}{max_{tilesmerge}} + \frac{boardscore_{new}-boardscore_{old}}{boardscore_{max}}
```

- **guide_score_reward**
  
This reward function was inspired by the human playing style that maintains the largest tile score at the corner, surrounded by tiles with incrementally lower scores. This play style aims to create a gradient across the board, which leads to better maximum scores. We calculated a score between [0,1] based on how well the board gradient is maintained after each stepping process. This score is then used as a bonus percentage calculated from the difference in board scores after stepping (similar to the full_score_reward calculation). The aim is to guide the agent to play in the same style as humans as described above.

```math
reward = (boardscore_{new} - boardscore_{old}) + (boardscore_{new} - boardscore_{old})*gradientscore
```

#### Scheduling

We also experimented with learning rate scheduling to compare which approach is most suitable with the selected reward terms. We conducted experiments using two training processes: one with a static learning rate of 0.00005, and another which scheduled the learning rate by decreasing its value throughout the training process until reaching the defined minimum learning rate value of approximately 0.000001.

#### Parameters

- `save_path`: Directory path where the trained model and logs will be saved
- `reward_func`: reward function used (`full_score_reward`, `merge_count_reward`, `guide_score_reward`)
- `n_episodes`: Total number of episodes for training
- `hidden_dim`: The number of hidden units in the neural network layers
- `target_update_interval`: Number of episodes between updates of the target network
- `initial_epsilon`: Initial value of epsilon in the epsilon-greedy algorithm
- `epsilon decay`: Decay factor applied to epsilon after each episode. Defines how fast it reduces over time
- `final_epsilon`: Minimum value of epsilon after full decay. Represents the exploitation threshold
- `learning_rate`: Step size for the optimizer during gradient updates
- `discount_factor`: Factor to discount future rewards
- `soft_update`: Boolean to enable soft update of target network
- `use_scheduler`: Boolean to enable scheduler
- `use_preset_board`: Boolean to enable preset_board
- `max_preset_tile`: Defines the highest tile value allowed during initialization
- `tau`: Soft update rate for the target network
- `batch_size`: Number of samples used per training update from the replay buffer
- `batch_size`: Maximum size of the replay buffer 


Example of YAML parameter settings for each experiment

```
DQN : 
  experiment_1:
    save_path : output/DQN_policy_exp1
    reward_func : Full_score
    n_episodes : 20000
    hidden_dim : 2048
    target_update_interval : 20
    initial_epsilon : 0.9
    epsilon_decay : 0.985
    final_epsilon : 0.01
    learning_rate : 0.00005
    discount_factor :  0.99
    soft_update : false
    use_scheduler : true
    use_preset_board : false
    max_preset_tile : 4
    tau : 0.95
    batch_size : 64
    buffer_size : 50000
```

#### Preset Board

We hypothesize that the initial states in 2048 are relatively easy to play, and that random actions can occasionally yield rewards. However, these early rewards may have limited value in enhancing the agent's learning. To address this, the agent is initialized from a pre-generated board where higher-value tiles, such as 512 or 1024, are already present, rather than starting from an empty or low-value board.

#### Performance Metrics

description description description


## Result & Analysis

description description description

#### 1. DQN 

description description description

#### 2. DQN

description description description

#### 3. DQN 

description description description

#### 4. Double DQN 

description description description

#### 5. Double DQN

description description description

#### 6. Dueling DQN 

description description description

#### 7. Dueling DQN 

description description description

#### 8. DQN

description description description

#### 9. DQN Skip

description description description

## Example Command

```
python scripts/train.py --algo DQN --exp experiment_1 --debug False
```

## Reference

- https://github.com/qwert12500/2048_rl
