##################################################
#
#     STANDARD IMPORT
#

from numpy import zeros, array, rot90
import random

class Board():
    def __init__(self):
        self.board = zeros((4, 4), dtype=int)
        self.fill_cell()
        self.game_over = False
        self.total_score = 0
    
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
                      j += 1
                      previous = None
                  else:
                      new_col[j] = previous
                      j += 1
                      previous = col[i]
      if previous != None:
          new_col[j] = previous
      return new_col

    def move(self, direction):
      # 0: left, 1: up, 2: right, 3: down
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

def main_loop(b, direction):
    new_board = b.move(direction)
    moved = False
    if (new_board == b.board).all():
        # move is invalid
        pass
    else:
        moved = True
        b.board = new_board
        b.fill_cell()
    return moved