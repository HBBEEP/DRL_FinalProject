import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Game_2048.board import Board, main_loop


# Sample Game (Manual) (Skip this cell if you dont want to try the game manually)
game = Board()
finish = False
while not finish:
  direction = int(input("please enter a direction: (0) Left, (1) Up, (2) Right, (3) Down"))
  if direction < 0 or direction > 3:
    print("Not a valid input! Please enter a valid value (0/1/2/3)!")
    continue
  moved = main_loop(game, direction)
  if not moved:
    print("Not a valid move! Nothing has changed.")
    continue
  print(game.board)
  print(game.total_score)
  finish = game.is_game_over()
print("Game Over!, Total Score is {}".format(game.total_score))