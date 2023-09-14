from collections import deque
from random import shuffle
import tensorflow as tf

import copy
import numpy as np

import settings as s

# Define Hyperparameter
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3
BOMB = -10

WINDOW_LENGTH = 7
INPUT_SHAPE = (4,7)

RETRAIN = False


# Define class template for network
# ATTENTION: Has to match the architecture from train.py
class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# Define rotation of movements based on agent position in the beginning
def get_movement_lookup(position):
    if position[0] != 1 and position[1] == 1:
        dir_lookup_RO = {'UP':'RIGHT',
                        'RIGHT':'DOWN',
                        'DOWN': 'LEFT',
                        'LEFT': 'UP',
                        'BOMB': 'BOMB',
                        'WAIT': 'WAIT'}
        return dir_lookup_RO
    
    elif position[0] == 1 and position[1] != 1:
        dir_lookup_LU = {'UP':'LEFT',
                        'RIGHT':'UP',
                        'DOWN': 'RIGHT',
                        'LEFT': 'DOWN',
                        'BOMB': 'BOMB',
                        'WAIT': 'WAIT'}
        return dir_lookup_LU
    
    elif position[0] != 1 and position[1] != 1:
        dir_lookup_RU = {'UP':'DOWN',
                        'RIGHT':'LEFT',
                        'DOWN': 'UP',
                        'LEFT': 'RIGHT',
                        'BOMB': 'BOMB',
                        'WAIT': 'WAIT'}
        return dir_lookup_RU
    
    elif position[0] == 1 and position[1] == 1:
        dir_lookup_LO = {'UP':'UP',
                        'RIGHT':'RIGHT',
                        'DOWN': 'DOWN',
                        'LEFT': 'UP',
                        'BOMB': 'BOMB',
                        'WAIT': 'WAIT'}
        return dir_lookup_LO

# Rotate map based on startposition of agent
def rotate_map(game_state_val, position):
    """" Coordinate based values:
        - field
        - bombs
        - explosion map
        - coins
        - self
        - others
    """
    game_state = copy.deepcopy(game_state_val)
    field_size = len(game_state['field'])-1 # Assume the map is quadratic
    
    if position == (1,1): # Remove later and make sure that function is only called in other cases
        return game_state
    
    elif position[0] != 1 and position[1] != 1:
        # Map based values
        game_state['field'] = [list(reversed(row)) for row in reversed(game_state['field'])]
        game_state['explosion_map'] = [list(reversed(row)) for row in reversed(game_state['explosion_map'])]

        # Coordinate based values
        game_state['bombs'] = [((field_size-o[0][0], field_size-o[0][1]), o[1]) for o in game_state['bombs']]
        game_state['coins'] = [(field_size-coord[0], field_size-coord[1]) for coord in game_state['coins']]
        game_state['self'] = (game_state['self'][0], game_state['self'][1], game_state['self'][2], (field_size-game_state['self'][3][0], field_size-game_state['self'][3][1]))
        game_state['others'] = [(o[0], o[1], o[2], (field_size-o[3][0], field_size-o[3][1])) for o in game_state['others']]

        return game_state


    elif position[0] == 1 and position[1] != 1:
        # Map based values
        game_state['field'] = list(reversed(list(map(list, zip(*game_state['field'])))))
        game_state['explosion_map'] = list(reversed(list(map(list, zip(*game_state['explosion_map'])))))

        # Coordinate based values
        # (x,y) -> y = x, x = len-y
        game_state['bombs'] = [((field_size-o[0][1], o[0][0]), o[1]) for o in game_state['bombs']]
        game_state['coins'] = [(field_size-coord[1], coord[0]) for coord in game_state['coins']]
        game_state['self'] = (game_state['self'][0], game_state['self'][1], game_state['self'][2], (field_size-game_state['self'][3][1], game_state['self'][3][0]))
        game_state['others'] = [(o[0], o[1], o[2], (field_size-o[3][1], o[3][0])) for o in game_state['others']]

        return game_state
    
    elif position[0] != 1 and position[1] == 1:
        
        # Map based values
        game_state['field'] = [list(reversed(row)) for row in zip(*game_state['field'])]
        game_state['explosion_map'] = [list(reversed(row)) for row in zip(*game_state['explosion_map'])]

        # Coordinate based values
        # (x,y) -> y = len-x, x = y
        game_state['bombs'] = [((o[0][1], field_size-o[0][0]), o[1]) for o in game_state['bombs']]
        game_state['coins'] = [(coord[1], field_size-coord[0]) for coord in game_state['coins']]
        game_state['self'] = (game_state['self'][0], game_state['self'][1], game_state['self'][2], (game_state['self'][3][1], field_size-game_state['self'][3][0]))
        game_state['others'] = [(o[0], o[1], o[2], (o[3][1], field_size-o[3][0])) for o in game_state['others']]

        return game_state
    

# Create input values for neural network by looking in every possible direction
def state_to_features(game_state, distance = WINDOW_LENGTH):
    """ Returns a 4 x distance matrix containing the events in the line of sight in every direction until distance """

    result = np.zeros((4, distance))
    x_self, y_self = game_state['self'][3]

    map = game_state['field']
    cols = len(map[0])
    rows = len(map)
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosion_map']

    for k, pos in enumerate([(-1,0),(1,0),(0,1),(0,-1)]):
        x_pos, y_pos = pos
        wall_bool = False

        for i in range(distance):
            # can't look further than the first wall
            if wall_bool == True:
                result[k,i] = WALL

            else:
                # Outside of map
                if x_self + x_pos*i < 0 or y_self + y_pos * i < 0 or x_self + x_pos*i > cols or y_self + y_pos*i > rows:
                    wall_bool = True
                    result[k,i] = WALL

                # wall
                elif map[x_self + x_pos*i][y_self + y_pos*i] == -1:
                    wall_bool = True
                    result[k,i] = WALL

                # Fill with free entries
                else:
                    result[k,i] = FREE

                    # bomb
                    # TODO: Think about weighting with countdown
                    for bo in bombs:
                        if bo[0][0] == x_self + x_pos*i and bo[0][1] == y_self + y_pos*i:
                            result[k,i] = BOMB

                    # coin
                    for co in coins:
                        if co[0] == x_self + x_pos*i and co[1] == y_self + y_pos*i:
                            result[k,i] = COIN

                    # explosion
                    if explosions[x_self + x_pos*i][y_self + y_pos*i] > 0:
                        result[k,i] = EXPLOSION

                    # crate
                    if map[x_self + x_pos*i][y_self + y_pos*i]:
                        result[k,i] = CRATE

    # Store bomb availability
    result[0,0] = game_state['self'][2]

    return result


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('BombiMcBombface resurrected!')
    np.random.seed()
    self.possible_actions = 6

    self.actions = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]

    self.current_round = 0
    self.orientation = 0
    self.next_action_adj = None
    self.lookup_move = None
    self.input_dim = INPUT_SHAPE[0] * INPUT_SHAPE[1]

    # Setup network
    model_weights_path = "q_network_weights.h5f"
    # If train and not retrain -> Create new network
    self.model = QNetwork(self.possible_actions)
    if self.train:
        if RETRAIN:
            self.model.load_weights(model_weights_path)

        # If RETRAIN is false: a new network is the correct way -> Do nothing
    else:
        # Load weights
        self.model.load_weights(model_weights_path)
    


def act(self, game_state):
    
    if game_state['step'] == 1: # First round, setup orientation
        self.orientation = game_state['self'][3]
        self.lookup_move = get_movement_lookup(game_state['self'][3])

    # Rotate Map
    game_state = rotate_map(game_state, self.orientation)

    current_orientation = game_state['self'][3]
    # Check if orientation correct
    self.logger.debug(f'The orientation should be (1,1) and is {current_orientation}')

    if self.train:
        if np.random.rand() < self.epsilon:
            index = np.random.randint(0, self.possible_actions - 1) # Explore
            prediction = index

        else:
            # Create state
            state = state_to_features(game_state).reshape(1,self.input_dim)
            self.old_gamestate = state
            
            # Predict
            prediction = self.model(state)
            prediction = np.argmax(prediction[0])


    else:
        # Create state
        state = state_to_features(game_state)
        self.old_gamestate = state
        
        # Predict
        prediction = self.model(state)
        prediction = np.argmax(prediction[0])

    # Resolve action

    action = self.actions[prediction]
    self.next_action_adj = action
    self.next_action = self.lookup_move[action]

    # Return action
    return self.next_action