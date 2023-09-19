import numpy as np


from . import hyperparameter as hp

def state_to_features(game_state):
    """ Returns a quadratic representation of the surroundings having the agent in the center
        Size: WINDOW_LENGTH x WINDOW_LENGTH
        WINDOW_LENGTH should be odd creating a field in the center
    """
    bomb_list = []

    result = np.zeros((hp.WINDOW_LENGTH, hp.WINDOW_LENGTH))
    
    x_self, y_self = game_state['self'][3]

    map = game_state['field']
    cols = len(map[0])
    rows = len(map)
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosion_map']
    others = game_state['others']


    halfdistance = int(np.floor(hp.WINDOW_LENGTH/2))
    # Iterate over fields necessary for the perspective
    for k, x_pos in enumerate(range(int(x_self)-halfdistance, int(x_self)+halfdistance+1)):
        for l, y_pos in enumerate(range(y_self - halfdistance, y_self + halfdistance +1)):
            result[k,l] = hp.FREE # Set field to free and overwrite if necessary
            if x_pos < 1 or x_pos >= rows-1 or y_pos < 1 or y_pos >= cols-1: # Outside the map or outer wall
                result[k,l] = hp.WALL

            else:
                for bo in bombs:
                    if bo[0][0] == x_pos and bo[0][1] == y_pos:
                        result[k,l] = hp.BOMB
                        bomb_list.append(((k,l), bo[1]))

                # coin
                for co in coins:
                    if co[0] == x_pos and co[1] == y_pos:
                        result[k,l] = hp.COIN

                # explosion
                if explosions[x_pos][y_pos] > 0:
                    result[k,l] = hp.EXPLOSION

                # crate
                if map[x_pos][y_pos] == 1:
                    result[k,l] = hp.CRATE

                # other agent
                for ot in others:
                    if ot[3][0] == x_pos and ot[3][1] == y_pos:
                        result[k,l] = hp.OTHER

    # Store the information about available bomb in the field of the own agent
    result[int(np.ceil(hp.WINDOW_LENGTH/2))][int(np.ceil(hp.WINDOW_LENGTH/2))] = game_state['self'][2]

    return result, bomb_list
