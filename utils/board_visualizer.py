import sys
import time
import numpy as np

class Board_Animator:
    def __init__(self):
        self.first = True

    def update(self, array,score, delay=0.0):
        # Clear the output by moving the cursor to the top and clearing the lines
        if not self.first:
            sys.stdout.write('\r\033[K')  # Clear the current line (only effective for single line)
            sys.stdout.flush()
            # Move back up to the beginning of the printed array (multiple lines)
            for _ in range(array.shape[0]+1):  # For each row in the array
                sys.stdout.write('\033[F')  # Move cursor up one line
                sys.stdout.write('\033[K')  # Clear that line
            sys.stdout.flush()

        self.first = False

        # Print the entire array (this will overwrite the previous content)
        print(array)
        print(f"board score : {score}")

        # Sleep to simulate delay
        time.sleep(delay)

    def done(self):
        """Move to the next line after the last update"""
        print()  # Move to the next line cleanly after updates
        self.first = True
        sys.stdout.flush()

