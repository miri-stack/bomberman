import copy
import os
from typing import List
import numpy as np
import events as e
import pickle
from .qmodel import SimpleQLearningAgent

# Hyperparameters
GAMMA = 0.95
EPSILON = 0.9
ALPHA = 0.3

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
    self.logger.info('QAgent is getting ready.')
    self.current_round = 0
    self.orientation = 0
    self.actions = ACTIONS
    self.state = None
    self.q_table = SimpleQLearningAgent()

    # Initialize Q-table
    self.q_table = SimpleQLearningAgent()  # Initialize your custom Q-table here

    # Check if a pre-trained Q-table exists and load it
    if not self.train and os.path.isfile("q_table.pickle"):
        self.logger.info("Loading Q-table from saved state.")
        with open("q_table.pickle", "rb") as file:
            loaded_q_table = pickle.load(file)
        self.q_table = loaded_q_table
    else:
        self.logger.info("Setting up model and Q-table from scratch.")
        # Initialize your Q-table here as needed
        self.q_table = SimpleQLearningAgent()  # Initialize Q-table


def act(self, game_state: dict):
    if game_state is not None:
        # Get a list of valid actions based on the current game state
        valid_actions = get_valid_actions(game_state, self.actions)
        
        # Add the 'valid_actions' key to the game_state dictionary
        game_state['valid_actions'] = valid_actions
        # Check if there are valid actions to choose from
        if valid_actions:
            # Convert the game state features to a tuple for Q-table lookup
            state_key = tuple(game_state_to_features(game_state))

            # Check if the state exists in the Q-table
            if state_key not in self.q_table.q_table:
                print("State not in Q-table")
                # Initialize the Q-values for this state if not present
                self.q_table.q_table[state_key] = {action: 0.0 for action in self.actions}

            # Modify the code to choose a single action based on your agent's logic
            if np.random.rand() < self.q_table.epsilon:
                selected_action = np.random.choice(valid_actions)  # Explore
            else:
                # Choose the action with the highest Q-value among valid actions
                q_values = {action: self.q_table.get_q_values(state_key, action) for action in valid_actions}
                selected_action = max(q_values, key=q_values.get)

            # Additional logic for improved decision-making
            nearest_coin = find_nearest_coin(game_state)
            hidden_coins = find_hidden_coins(game_state)
            opponent_action = hunt_opponents(game_state)
            battle_action = battle_opposing_agents(game_state)

            if nearest_coin:
                # If there's a nearest coin, prioritize collecting it
                coin_x, coin_y = nearest_coin

                # Get the agent's orientation (dx, dy) from game_state
                orientation = game_state['self'][3]

                # Calculate the agent's current position
                self_x, self_y = game_state['self'][3]

                # Use the orientation to determine the action lookup
                action_lookup = get_movement_lookup(orientation)

                if (coin_x, coin_y) in hidden_coins:
                    # If the nearest coin is hidden, consider dropping a bomb
                    if action_lookup[selected_action] != "BOMB" and action_lookup[selected_action] != "WAIT":
                        selected_action = "BOMB"
                else:
                    # Move towards the nearest visible coin
                    target_direction = get_direction_towards(self_x, self_y, coin_x, coin_y)
                    if target_direction in valid_actions:
                        selected_action = target_direction

            if opponent_action:
                # If there's an opponent nearby, consider the opponent's suggested action
                if opponent_action in valid_actions:
                    selected_action = opponent_action

            if battle_action:
                # If in a battle situation, follow the battle action
                if battle_action in valid_actions:
                    selected_action = battle_action

            return selected_action
    
    # If no valid actions are available, return a default action
    return "WAIT"


def game_state_to_features(game_state):
    # Initialize a 21x21 grid centered around the agent
    grid_size = 5
    features = np.zeros((grid_size, grid_size))

    # Extract relevant information from game_state
    field = game_state['field']
    self_position = game_state['self'][3]
    bombs = game_state['bombs'] # TODO: run away from the first detonation 
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    orientation = game_state['self'][3]  # Agent's orientation (0, 1, 2, or 3) TODO: use johannas orientation

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

def get_movement_lookup(position):
    if position[0] != 1 and position[1] == 1:
        dir_lookup_RO = {
            'UP': 'RIGHT',
            'RIGHT': 'DOWN',
            'DOWN': 'LEFT',
            'LEFT': 'UP',
            'BOMB': 'BOMB',
            'WAIT': 'WAIT'
        }
        return dir_lookup_RO
    elif position[0] == 1 and position[1] != 1:
        dir_lookup_LU = {
            'UP': 'LEFT',
            'RIGHT': 'UP',
            'DOWN': 'RIGHT',
            'LEFT': 'DOWN',
            'BOMB': 'BOMB',
            'WAIT': 'WAIT'
        }
        return dir_lookup_LU
    elif position[0] != 1 and position[1] != 1:
        dir_lookup_RU = {
            'UP': 'DOWN',
            'RIGHT': 'LEFT',
            'DOWN': 'UP',
            'LEFT': 'RIGHT',
            'BOMB': 'BOMB',
            'WAIT': 'WAIT'
        }
        return dir_lookup_RU
    elif position[0] == 1 and position[1] == 1:
        dir_lookup_LO = {
            'UP': 'UP',
            'RIGHT': 'RIGHT',
            'DOWN': 'DOWN',
            'LEFT': 'UP',
            'BOMB': 'BOMB',
            'WAIT': 'WAIT'
        }
        return dir_lookup_LO

def rotate_map(game_state_val, position):
    game_state = copy.deepcopy(game_state_val)
    field_size = len(game_state['field']) - 1  # Assume the map is quadratic

    if position == (1, 1):  # No rotation needed for (1, 1) orientation
        return game_state

    elif position[0] != 1 and position[1] != 1:
        # Rotate map-based values
        game_state['field'] = [list(reversed(row)) for row in reversed(game_state['field'])]
        game_state['explosion_map'] = [list(reversed(row)) for row in reversed(game_state['explosion_map'])]

        # Rotate coordinate-based values
        game_state['bombs'] = [((field_size - o[0][0], field_size - o[0][1]), o[1]) for o in game_state['bombs']]
        game_state['coins'] = [(field_size - coord[0], field_size - coord[1]) for coord in game_state['coins']]
        game_state['self'] = (
            game_state['self'][0], game_state['self'][1], game_state['self'][2],
            (field_size - game_state['self'][3][0], field_size - game_state['self'][3][1]))
        game_state['others'] = [(o[0], o[1], o[2], (field_size - o[3][0], field_size - o[3][1])) for o in
                                game_state['others']]

        return game_state

    elif position[0] == 1 and position[1] != 1:
        # Rotate map-based values
        game_state['field'] = list(reversed(list(map(list, zip(*game_state['field'])))))
        game_state['explosion_map'] = list(reversed(list(map(list, zip(*game_state['explosion_map'])))))

        # Rotate coordinate-based values
        game_state['bombs'] = [((field_size - o[0][1], o[0][0]), o[1]) for o in game_state['bombs']]
        game_state['coins'] = [(field_size - coord[1], coord[0]) for coord in game_state['coins']]
        game_state['self'] = (
            game_state['self'][0], game_state['self'][1], game_state['self'][2],
            (field_size - game_state['self'][3][1], game_state['self'][3][0]))
        game_state['others'] = [(o[0], o[1], o[2], (field_size - o[3][1], o[3][0])) for o in game_state['others']]

        return game_state

    elif position[0] != 1 and position[1] == 1:
        # Rotate map-based values
        game_state['field'] = [list(reversed(row)) for row in zip(*game_state['field'])]
        game_state['explosion_map'] = [list(reversed(row)) for row in zip(*game_state['explosion_map'])]

        # Rotate coordinate-based values
        game_state['bombs'] = [((o[0][1], field_size - o[0][0]), o[1]) for o in game_state['bombs']]
        game_state['coins'] = [(coord[1], field_size - coord[0]) for coord in game_state['coins']]
        game_state['self'] = (
            game_state['self'][0], game_state['self'][1], game_state['self'][2],
            (game_state['self'][3][1], field_size - game_state['self'][3][0]))
        game_state['others'] = [(o[0], o[1], o[2], (o[3][1], field_size - o[3][0])) for o in game_state['others']]

        return game_state

def get_directions():
    # Return possible movement directions (UP, DOWN, LEFT, RIGHT)
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    return directions

def get_valid_actions(game_state: dict, actions: List[str]):
    valid_actions = []

    # Define a helper function to check if a position is valid
    def is_valid_position(x, y):
        if (
            0 <= x < len(game_state['field'][0])
            and 0 <= y < len(game_state['field'])
            and game_state['field'][y][x] not in {WALL, CRATE}
        ):
            return True
        return False

    # Get the agent's current position
    x, y = game_state['self'][3]

    for action in actions:
        if action == 'UP':
            new_x, new_y = x, y - 1
        elif action == 'DOWN':
            new_x, new_y = x, y + 1
        elif action == 'LEFT':
            new_x, new_y = x - 1, y
        elif action == 'RIGHT':
            new_x, new_y = x + 1, y
        else:
            new_x, new_y = x, y  # For 'WAIT' and 'BOMB' actions

        # Check if the new position is valid (not a wall or crate)
        if is_valid_position(new_x, new_y):
            valid_actions.append(action)

    # Avoid moving towards bombs that are about to explode
    bomb_positions = {bomb[0] for bomb in game_state['bombs']}
    safe_actions = []
    for action in valid_actions:
        dx, dy = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}.get(action, (0, 0))
        new_x, new_y = x + dx, y + dy
        if (new_x, new_y) not in bomb_positions:
            safe_actions.append(action)

    # Ensure that the agent doesn't move into a cell occupied by another player or itself
    occupied_positions = {(p[3][0], p[3][1]) for p in game_state['others']}
    occupied_positions.add((x, y))  # Add the agent's current position

    # If there are safe actions, prioritize them over 'WAIT'
    if safe_actions:
        return safe_actions

    # If there are no safe actions, return 'WAIT'
    return ['WAIT']



def get_direction_towards(start_x, start_y, target_x, target_y):
    """
    Calculate the direction from start position to target position.

    :param start_x: X-coordinate of the starting position.
    :param start_y: Y-coordinate of the starting position.
    :param target_x: X-coordinate of the target position.
    :param target_y: Y-coordinate of the target position.
    :return: A string representing the direction ('UP', 'DOWN', 'LEFT', 'RIGHT') to move from start to target.
    """
    if start_x < target_x:
        return 'RIGHT'
    elif start_x > target_x:
        return 'LEFT'
    elif start_y < target_y:
        return 'DOWN'
    elif start_y > target_y:
        return 'UP'
    else:
        return 'WAIT'

def move_in_direction(game_state, direction):
    x, y = game_state['self'][3]  # Current position
    if direction == 'UP':
        new_x, new_y = x, y - 1
    elif direction == 'DOWN':
        new_x, new_y = x, y + 1
    elif direction == 'LEFT':
        new_x, new_y = x - 1, y
    elif direction == 'RIGHT':
        new_x, new_y = x + 1, y
    else:
        new_x, new_y = x, y  # Default to the current position for non-movement actions
    return new_x, new_y

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
            # If opponent is within a safe distance, prioritize attacking them with bombs
            # Calculate the direction to the opponent
            direction = get_direction_towards(self_x, self_y, opponent[0], opponent[1])
            return direction  # Return the direction to attack the opponent
    
    # No nearby opponents, return None
    return None

def battle_opposing_agents(game_state):
    # Engage and compete against opposing agents based on the game state
    opponents = [player[3] for player in game_state['others']]
    self_x, self_y = game_state['self'][3]

    # Determine a safe distance to engage opponents
    safe_distance = 2  # Adjust as needed

    for opponent in opponents:
        if manhattan_distance((self_x, self_y), opponent) <= safe_distance:
            # If opponent is within a safe distance, prioritize attacking them with bombs
            # Calculate the direction to the opponent
            direction = get_direction_towards(self_x, self_y, opponent[0], opponent[1])
            return direction  # Return the direction to attack the opponent
    
    # No nearby opponents, return None
    return None

def manhattan_distance(pos1, pos2):
    # Calculate the Manhattan distance between two positions
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)
