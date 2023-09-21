from collections import deque, defaultdict
import os
import numpy as np
import random
import events as e
import pickle
from typing import List

# Hyperparameters
GAMMA = 0.9
EPSILON = 0.1
ALPHA = 0.1

# Constants for feature values
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3
BOMB = -10

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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

    def save_q_table(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(dict(self.q_table), file)

    def load_q_table(self, file_name):
        with open(file_name, 'rb') as file:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(file))

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)



def game_state_to_features(game_state):
    # Initialize a 21x21 grid centered around the agent
    grid_size = 21
    features = np.zeros((grid_size, grid_size))

    # Extract relevant information from game_state
    field = game_state['field']
    self_position = game_state['self'][3]
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    orientation = game_state['self'][2]  # Agent's orientation (0, 1, 2, or 3)

    # Calculate the position of the agent in the centered grid
    agent_x, agent_y = self_position
    grid_center = grid_size // 2
    x_offset = grid_center - agent_x
    y_offset = grid_center - agent_y

    # Populate the centered grid with features based on game state
    for x in range(grid_size):
        for y in range(grid_size):
            x_pos, y_pos = agent_x + x - grid_center, agent_y + y - grid_center

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

    # Flatten the centered grid into a 2D feature vector
    return features.reshape(-1)

def get_directions():
    # Return possible movement directions (UP, DOWN, LEFT, RIGHT)
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    return directions

def get_valid_moves(game_state):
    # Return valid movements based on the game state
    valid_moves = []
    directions = get_directions()
    
    # Check if the move in the given direction is valid
    for direction in directions:
        new_x, new_y = move_in_direction(game_state, direction)
        if is_valid_position(game_state, new_x, new_y):
            valid_moves.append(direction)
            
    return valid_moves

def move_in_direction(game_state, direction):
    # Calculate the new position when moving in the given direction
    x, y = game_state['self'][3]
    if direction == "UP":
        x -= 1
    elif direction == "DOWN":
        x += 1
    elif direction == "LEFT":
        y -= 1
    elif direction == "RIGHT":
        y += 1
    return x, y

def is_valid_position(game_state, x, y):
    # Check if the given position is valid (not a wall or out of bounds)
    field = game_state['field']
    if 0 <= x < len(field) and 0 <= y < len(field[0]) and field[x][y] != -1:
        return True
    return False

def find_nearest_coin(game_state):
    # Find the nearest coin based on the game state
    coins = game_state['coins']
    if coins:
        self_x, self_y = game_state['self'][3]
        nearest_coin = min(coins, key=lambda c: manhattan_distance((self_x, self_y), c))
        return nearest_coin
    return None

def find_hidden_coins(game_state):
    # Find all hidden coins based on the game state
    visible_coins = set(game_state['coins'])
    all_possible_positions = [(x, y) for x in range(21) for y in range(21)]
    hidden_coins = [pos for pos in all_possible_positions if pos not in visible_coins]
    return hidden_coins

def hunt_opponents(game_state):
    # Hunt and blow up opponents based on the game state
    opponents = [player[3] for player in game_state['others']]
    bombs = game_state['bombs']
    self_x, self_y = game_state['self'][3]

    # Determine a safe distance to engage opponents
    safe_distance = 2  # Adjust as needed

    for opponent in opponents:
        if manhattan_distance((self_x, self_y), opponent) <= safe_distance:
            # If opponent is within a safe distance, drop a bomb to engage
            return "BOMB"
    
    # No need to engage opponents, return None
    return None

def battle_opposing_agents(game_state):
    # Engage and compete against opposing agents based on the game state
    # You can use more advanced strategies to compete with opponents
    # Implement logic to compete with opposing agents
    # Return an action to compete or None if no action is needed
    
    # For example, if there is an opponent within range, drop a bomb to engage
    opponents = [player[3] for player in game_state['others']]
    self_x, self_y = game_state['self'][3]

    # Determine a safe distance to engage opponents
    safe_distance = 2  # Adjust as needed

    for opponent in opponents:
        if manhattan_distance((self_x, self_y), opponent) <= safe_distance:
            # If opponent is within a safe distance, drop a bomb to engage
            return "BOMB"
    
    # No need to engage opponents, return None
    return None

def manhattan_distance(pos1, pos2):
    # Calculate the Manhattan distance between two positions
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)
