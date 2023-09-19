from collections import deque
from random import shuffle
import tensorflow as tf

import copy
import numpy as np
from . import hyperparameter as hp

import settings as s
from .orientation import get_movement_lookup, rotate_map
from .perspective import state_to_features


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

    self.current_round = 0
    self.orientation = 0
    self.next_action_adj = None
    self.lookup_move = None
    self.input_dim = hp.INPUT_SHAPE[0] * hp.INPUT_SHAPE[1]

    # Setup network
    model_weights_path = "q_network_weights.h5f"
    # If train and not retrain -> Create new network
    self.model = QNetwork(self.possible_actions)
    dummy_input = np.full((1,hp.INPUT_SHAPE[0]*hp.INPUT_SHAPE[1]),1)  # Adjust the input shape as needed
    self.model(dummy_input)

    if self.train:
        if hp.RETRAIN:
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
    if game_state['step'] == 1:
        self.logger.debug(f'The orientation should be (1,1) and is {current_orientation}')

    if self.train:
        if np.random.rand() < self.epsilon:
            index = np.random.randint(0, self.possible_actions - 1) # Explore
            prediction = index

        else:
            # Create state
            state = state_to_features(game_state)[0].reshape(1,self.input_dim)
            self.old_gamestate = state
            
            # Predict
            prediction = self.model(state)
            prediction = np.argmax(prediction[0])


    else:
        # Create state
        state, _ = state_to_features(game_state)[0].reshape(1,self.input_dim)
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