import numpy as np


from . import hyperparameter as hp

# Create input values for neural network by looking in every possible direction
def state_to_features(game_state, distance = hp.WINDOW_LENGTH):
    """ Returns a 4 x distance matrix containing the events in the line of sight in every direction until distance """

    result = np.zeros((4, distance))
    x_self, y_self = game_state['self'][3]

    map = game_state['field']
    cols = len(map[0])
    rows = len(map)
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosion_map']
    others = game_state['others']

    for k, pos in enumerate([(-1,0),(1,0),(0,1),(0,-1)]):
        x_pos, y_pos = pos
        WALL_bool = False

        for i in range(distance):
            # can't look further than the first hp.WALL
            if WALL_bool == True:
                result[k,i] = hp.WALL

            else:
                # Outside of map
                if x_self + x_pos*i < 0 or y_self + y_pos * i < 0 or x_self + x_pos*i > cols or y_self + y_pos*i > rows:
                    WALL_bool = True
                    result[k,i] = hp.WALL

                # hp.WALL
                elif map[x_self + x_pos*i][y_self + y_pos*i] == -1:
                    WALL_bool = True
                    result[k,i] = hp.WALL

                # Fill with FREE entries
                else:
                    result[k,i] = hp.FREE

                    # bomb
                    for bo in bombs:
                        if bo[0][0] == x_self + x_pos*i and bo[0][1] == y_self + y_pos*i:
                            result[k,i] = hp.BOMB

                    # coin
                    for co in coins:
                        if co[0] == x_self + x_pos*i and co[1] == y_self + y_pos*i:
                            result[k,i] = hp.COIN

                    # explosion
                    if explosions[x_self + x_pos*i][y_self + y_pos*i] > 0:
                        result[k,i] = hp.EXPLOSION

                    # crate
                    if map[x_self + x_pos*i][y_self + y_pos*i] == 1:
                        result[k,i] = hp.CRATE

                    # other players
                    for agent in others:
                        if agent[3][0] == x_self + x_pos*i and agent[3][1] == y_self + y_pos*i:
                            result[k,i] = hp.PLAYER

    # Store bomb availability
    result[0,0] = game_state['self'][2]

    return result