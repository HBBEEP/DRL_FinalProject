##################################################
#
#     STANDARD IMPORT
#

import numpy as np
from numpy import zeros, array, rot90
import random

class Board():
    def __init__(self):
        self.board = zeros((4, 4), dtype=int)
        self.fill_cell()
        self.game_over = False
        self.total_score = 0
        self.tile_merge = 0

    def set_max_tile(self, max_tile:int):
        """Preset the board with a given maximum tile (256, 512, 1024)."""
        assert max_tile in [256, 512, 1024], "Only 256, 512, or 1024 allowed."

        self.max_tile = max_tile

    def reset(self):
        self.__init__()
    
    # Adding a random 2/4 into the board
    def fill_cell(self):
      i, j = (self.board == 0).nonzero()
      if i.size != 0:
          rnd = random.randint(0, i.size - 1) 
          self.board[i[rnd], j[rnd]] = 2 * ((random.random() > .9) + 1)
    
    # Moving tiles in a column to left and merge if possible
    def move_left(self, col):
        new_col = zeros((4), dtype=col.dtype)
        j = 0
        previous = None
        for i in range(col.size):
            if col[i] != 0: # number different from zero
                if previous == None:
                    previous = col[i]
                else:
                    if previous == col[i]:
                        new_col[j] = 2 * col[i]
                        self.total_score += new_col[j]
                        self.tile_merge += 2
                        j += 1
                        previous = None
                    else:
                        new_col[j] = previous
                        j += 1
                        previous = col[i]
        if previous != None:
            new_col[j] = previous
        return new_col

    def preset_board(self):
        self.board = np.zeros((4, 4), dtype=int)
        
        # Place the max_tile in a random cell
        empty_cells = [(i, j) for i in range(4) for j in range(4)]
        random.shuffle(empty_cells)
        max_tile_pos = empty_cells.pop()
        self.board[max_tile_pos] = self.max_tile
        
        # Fill a few other cells with random values < max_tile
        for _ in range(random.randint(3, 6)):
            if not empty_cells:
                break
            i, j = empty_cells.pop()
            
            # Choose a random tile that is a power of two and < max_tile
            possible_tiles = [2**i for i in range(1, int(np.log2(self.max_tile)))]
            self.board[i, j] = random.choice(possible_tiles)

    def move(self, direction):
      # 0: left, 1: up, 2: right, 3: down
      self.tile_merge = 0
      rotated_board = rot90(self.board, direction)
      cols = [rotated_board[i, :] for i in range(4)]
      new_board = array([self.move_left(col) for col in cols])
      return rot90(new_board, -direction)
    
    def is_game_over(self):
      for i in range(self.board.shape[0]):
        for j in range(self.board.shape[1]):
          if self.board[i][j] == 0:
            return False
          if i != 0 and self.board[i - 1][j] == self.board[i][j]:
            return False
          if j != 0 and self.board[i][j - 1] == self.board[i][j]:
            return False
      return True

    def step(self, direction):
      new_board = self.move(direction)
      if not (new_board == self.board).all():
        self.board = new_board
        self.fill_cell()

def main_loop(b:Board, direction):
    new_board = b.move(direction)
    moved = False
    if (new_board == b.board).all():
        pass
    else:
        moved = True
        b.board = new_board
        b.fill_cell()
    return moved

if __name__ == "__main__":
  b = Board()
  b.set_max_tile(512)
  b.preset_board()
  print(b.board)
