from collections import deque, defaultdict
import numpy as np
import random
import events as e
from .train import reward_from_events

# Hyperparameters
GAMMA = 0.9
EPSILON = 0.1
ALPHA = 0.1

class SimpleQLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state = None
        self.action = None
        self.epsilon = EPSILON
        self.alpha = ALPHA
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]

    def act(self, game_state):
        if game_state is not None:
            self.state = tuple(game_state_to_features(game_state))
            if np.random.rand() < self.epsilon:
                self.action = np.random.choice(self.actions)  # Explore
            else:
                # Choose the action with the highest Q-value
                q_values = self.q_table[self.state]
                if q_values:
                    self.action = max(q_values, key=q_values.get)
                else:
                    self.action = np.random.choice(self.actions)  # If no Q-values are available, explore

        return self.action

    def update_q_table(self, old_state, new_state, action, reward):
        if old_state is not None and new_state is not None:
            max_q_value_new_state = max(self.q_table[new_state].values(), default=0.0)
            q_value_old_state_action = self.q_table[old_state][action]
            self.q_table[old_state][action] = q_value_old_state_action + \
                                              self.alpha * (reward + GAMMA * max_q_value_new_state - q_value_old_state_action)

def setup(self):
    self.logger.info('Creating a Simple Q-learning agent')
    self.agent = SimpleQLearningAgent()

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is not None:
        old_state = tuple(game_state_to_features(old_game_state))
        new_state = tuple(game_state_to_features(new_game_state))
        reward = reward_from_events(self, events)
        self.agent.update_q_table(old_state, new_state, self_action, reward)

def end_of_round(self, last_game_state, last_action, events):
    pass

def game_state_to_features(game_state):
    features = np.zeros((4, 7))  # Initialize a 4x7 grid for features

    # Extract relevant information from game_state
    field = game_state['field']
    self_position = game_state['self'][3]
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']

    # Define constants for feature values
    WALL = -1
    EXPLOSION = -3
    FREE = 0
    CRATE = 1
    COIN = 2
    PLAYER = 3
    BOMB = -10

    # Define a mapping of event types to feature values
    event_to_feature = {
        e.BOMB_EXPLODED: EXPLOSION,
        e.COIN_COLLECTED: COIN,
        e.CRATE_DESTROYED: FREE,
        e.KILLED_OPPONENT: PLAYER,
        e.KILLED_SELF: PLAYER,
        e.WAITED: FREE,
        e.BOMB_DROPPED: BOMB,
        e.SURVIVED_ROUND: FREE,
    }

    # Iterate over the 4x7 grid and populate features
    for x in range(4):
        for y in range(7):
            x_pos, y_pos = self_position[0] - (x - 1), self_position[1] - (y - 3)

            if x_pos < 0 or y_pos < 0 or x_pos >= len(field) or y_pos >= len(field[0]):
                # Out of bounds, consider it a wall
                features[x, y] = WALL
            else:
                cell_value = field[x_pos][y_pos]
                # Map the cell value to the corresponding feature
                if cell_value == -1:  # Wall
                    features[x, y] = WALL
                elif cell_value == 1:  # Crate
                    features[x, y] = CRATE
                elif cell_value == 0:  # Free space
                    features[x, y] = FREE
                elif cell_value == 3:  # Player
                    features[x, y] = PLAYER

                # Check for coins at the current position
                if (x_pos, y_pos) in coins:
                    features[x, y] = COIN

                # Check for bombs at the current position
                for (bomb_x, bomb_y), _ in bombs:
                    if x_pos == bomb_x and y_pos == bomb_y:
                        features[x, y] = BOMB

                # Check for explosion at the current position
                if explosion_map[x_pos][y_pos] > 0:
                    features[x, y] = EXPLOSION

    # Flatten the 4x7 grid into a 28-dimensional feature vector
    return features.reshape(-1)
