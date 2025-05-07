import torch
import numpy as np

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


def guide_score_reward(board_total_score, old_score, new_board, old_board, device='cuda'):
    reward = (board_total_score - old_score)
    guide_score = (gradient_score_2048(new_board) - gradient_score_2048(old_board))
    reward = reward + reward*guide_score
    reward = torch.tensor([reward], dtype=torch.float, device=device)
    return reward



def gradient_score_2048(board):
    assert board.shape == (4, 4), "Input must be a 4x4 array."

    corners = {
        'top-left': (0, 0),
        'top-right': (0, 3),
        'bottom-left': (3, 0),
        'bottom-right': (3, 3)
    }

    max_val = np.max(board)
    max_pos = tuple(np.argwhere(board == max_val)[0])

    # Choose the *closest* corner to max_pos
    distances = {name: np.linalg.norm(np.array(pos) - np.array(max_pos)) for name, pos in corners.items()}
    best_corner = min(distances, key=distances.get)
    min_distance = distances[best_corner]

    # Normalize distance into a weight: 1 if in corner, decreases as it gets further
    max_possible_distance = np.linalg.norm([3, 3])
    corner_weight = 1.0 - (min_distance / max_possible_distance)

    # Create a distance map from the chosen corner
    corner_pos = corners[best_corner]
    dist_map = np.fromfunction(
        lambda i, j: np.sqrt((i - corner_pos[0])**2 + (j - corner_pos[1])**2),
        (4, 4)
    )

    # Flatten arrays
    dist_flat = dist_map.flatten()
    board_flat = board.flatten()

    # Sort board values by proximity to the corner
    sorted_idx = np.argsort(dist_flat)
    sorted_vals_by_dist = board_flat[sorted_idx]

    # Ideal: descending sorted values
    ideal = np.sort(board_flat)[::-1]

    match_count = np.sum([1 for i in range(len(ideal)) if ideal[i] == sorted_vals_by_dist[i]])
    final_score = match_count/16
    return final_score
