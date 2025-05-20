# Comparative DQN Approaches for 2048 Puzzle Game
FRA 503: Deep Reinforcement Learning for Robotics

## Member
1. Kullakant Kaewkallaya 64340500006
2. Peerawat Santifuengkul 64340500043
3. Monsicha Sopitlaptana 64340500071


## Introduction

2048 is a sliding tile puzzle game. The objective of the game is to slide numbered tiles on a grid to combine them and create a tile with the number 2048. This game is played on a plain 4×4 grid, with numbered tiles that slide when a player moves them using the four arrow keys. 

In this project, we will implement Deep Reinforcement Learning to play the 2048 game using different algorithms: Deep Q-Network, Double Deep Q-Network, and Dueling Deep Q-Network to compare the performance of each algorithm, including differences in reward functions and experimental setups.

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

#### Parameters

description description description

#### Reward Functions

description description description

#### Scheduling

description description description

#### Time Skip

description description description

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
