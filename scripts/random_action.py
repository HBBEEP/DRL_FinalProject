import matplotlib.pyplot as plt
import numpy as np
import math

from Game_2048.board import Board, main_loop

def sample_game_random():
  game = Board()
  finish = False
  while not finish:
    direction = np.random.randint(4)
    moved = main_loop(game, direction)
    if not moved:
      # Sample another direction if the move is invalid
      continue
    finish = game.is_game_over()
  total_score = game.total_score
  best_tile = game.board.max()
  return total_score, best_tile

def main():
    scores_random, best_tiles_random = [], []
    for i in range(1000):
        if i % 100 == 0:
            print(f"Iteration {i}")
        total_score, best_tile = sample_game_random()
        scores_random.append(total_score)
        best_tiles_random.append(best_tile)
    print("Finish")

    # Plot best score distribution
    plt.hist(scores_random, bins = 50)
    plt.title("Total score distribution")
    plt.xlabel("Total Score")
    plt.ylabel("Frequency")
    plt.show()

    # Plot best score distribution
    max_power = int(math.log(max(best_tiles_random), 2)) + 1
    min_power = int(math.log(min(best_tiles_random), 2))
    unique, counts = np.unique(best_tiles_random, return_counts=True)
    plt.bar([str(2 ** i) for i in range(min_power, max_power)], counts)
    plt.title("Best tile distribution")
    plt.xlabel("Best tile")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
   main()