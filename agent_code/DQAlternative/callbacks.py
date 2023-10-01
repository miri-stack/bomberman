from collections import deque
from random import shuffle
import tensorflow as tf

import copy
import numpy as np
from . import hyperparameter as hp

import settings as s
from .orientation import get_movement_lookup, rotate_map, rotate_other_values
from .perspective import state_to_features

# Enable the following for simple playing due to a bug in tf
#tf.config.experimental.set_visible_devices([], 'GPU')


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

    self.orientation = 0
    self.stepcount = 0
    self.next_action_adj = None
    self.lookup_move = None
    self.field = np.zeros((17,17))
    self.input_dim = 4 * 6 + 2

    # Setup network
    model_weights_path = "models/q_network_weights.h5"
    # If train and not retrain -> Create new network
    self.model = QNetwork(self.possible_actions)
    dummy_input = np.full((1,self.input_dim),1)  # Adjust the input shape as needed
    self.model(dummy_input)

    if not self.train or hp.RETRAIN:
        # Load weights
        self.model.load_weights(model_weights_path)

    # Else do nothing


def act(self, game_state):
    
    if game_state['step'] == 1: # First round, setup orientation
        self.orientation = game_state['self'][3]
        self.lookup_move = get_movement_lookup(game_state['self'][3])

        
    # Rotate Map
    self.field = rotate_map(game_state, self.orientation)
    game_state = rotate_other_values(game_state, self.orientation)
    
    if self.train and np.random.rand() < self.epsilon:
        # Define the probabilities for each index
        probabilities = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

        # Generate a random integer between 0 and 5 with the specified probabilities
        prediction = np.random.choice(range(6), p=probabilities)

    else:
        # Create state
        state = state_to_features(self.field, game_state)
        self.old_gamestate = state
        
        # Predict
        prediction = self.model(state.reshape(1,self.input_dim))
        prediction = np.argmax(prediction[0])

    # Resolve action

    action = self.actions[prediction]
    self.next_action_adj = action
    self.next_action = self.lookup_move[action]
    #self.logger.debug(f'Chosen move: {self.next_action}')
    self.stepcount += 1

    # Return action
    return self.next_action