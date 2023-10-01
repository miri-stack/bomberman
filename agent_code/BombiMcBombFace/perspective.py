from collections import deque
import numpy as np
from . import hyperparameter as hp

def get_manhattan_dist(koord1, koord2):
    # Compute the manhattan distance
    return np.abs(koord1[0]-koord2[0]) + np.abs(koord1[1]-koord2[1])

halfdistance = int(np.floor(hp.WINDOW_LENGTH/2))

# Create input values for neural network by looking in every possible direction
def state_to_features(field, game_state):
    """ Returns a quadratic representation of the surroundings having the agent in the center
        Size: WINDOW_LENGTH x WINDOW_LENGTH
        WINDOW_LENGTH should be odd creating a field in the center
    """
    result = np.full((21,21),hp.WALL) # Place old map in bigger version

    # Place crates and walls
    result[2:19,2:19] = np.array(field, dtype=float)

    
    x_self, y_self = game_state['self'][3]
    x_self = x_self + 2
    y_self = y_self + 2

    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosion_map']

    # Find the edges of the perspective in the original field.
    x_min, x_max = int(x_self) - halfdistance, int(x_self) + halfdistance
    y_min, y_max = int(y_self) - halfdistance, int(y_self) + halfdistance


    # Place coin information in result matrix
    for co in coins:
        result[co[0]+2][co[1]+2] += hp.COIN


    # Place bomb information in result matrix
    for bo in bombs:
        for move in [(0,1),(1,0),(-1,0),(0,-1)]:
            for i in range(4):
                if result[bo[0][0]+2+i*move[0]][bo[0][1]+2+i*move[1]] == hp.WALL: # Only add explosion when no wall to protect
                    break
                else:
                    result[bo[0][0]+2 + i*move[0], bo[0][1]+2+ i*move[1]] += hp.BOMB / (bo[1] + 1)

    # Place explosion information in result matrix
    explosions = np.array(explosions, dtype=float)
    explosions[explosions != 0] = hp.BOMB
    result[2:19,2:19] += explosions


    # Cast bomb_bool to float32
    bomb_bool = np.float32(game_state['self'][2])

    # Check if other actions are available and cast to float32
    right_bool = np.float32(result[x_self + 1][y_self] <= 0)
    left_bool = np.float32(result[x_self - 1][y_self] <= 0)
    up_bool = np.float32(result[x_self][y_self - 1] <= 0)
    down_bool = np.float32(result[x_self][y_self + 1] <= 0)

    # Choose correct part of field
    result = result[x_min:x_max+1,y_min:y_max+1]

    # Create the result array without reshaping it multiple times
    result = np.zeros((1, hp.WINDOW_LENGTH * hp.WINDOW_LENGTH + 6), dtype=np.float32)

    # Set the values in the result array
    result[0, -6:] = [up_bool, down_bool, left_bool, right_bool, bomb_bool, 1.0]


    return result

    