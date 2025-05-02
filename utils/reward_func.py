

def normal_reward(board_total_score, old_score, tile_merge, goal=2048, tile_divider=8):
    move_reward = (board_total_score - old_score)/goal
    tile_merge_reward = (tile_merge-1)/tile_divider
    reward = tile_merge_reward + move_reward
    return reward