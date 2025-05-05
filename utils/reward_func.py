import torch

def full_score_reward(board_total_score, old_score, device='cuda'):
    reward = (board_total_score - old_score)
    reward = torch.tensor([reward], device=device)
    return reward


def merge_count_reward(board_total_score, old_score, tile_merge, goal=2048, tile_divider=8, device='cuda'):
    move_reward = (board_total_score - old_score)/goal
    tile_merge_reward = (tile_merge-1)/tile_divider
    reward = tile_merge_reward + move_reward

    reward = torch.tensor([reward], device=device)
    return reward