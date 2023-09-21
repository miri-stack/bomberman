import numpy as np
import pickle, os
import events as e


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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

def setup(self):
    self.logger.info('Mary Shelly is getting ready.')
    self.current_round = 0
    self.orientation = 0

    if self.train and not os.path.isfile('my-saved-model.pt'):
        self.logger.info('Setting up model from scratch.')
        self.qtable = {}
    else:
        self.logger.info('Loading model from saved state.')
        with open('my-saved-model.pt', 'rb') as file:
            self.qtable = pickle.load(file)


    # initialize Q(S,A) to 0 for all S and A
    # Start Field 


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    # observe environment

    if game_state['step'] == 1: # First round, setup orientation
        self.orientation = game_state['self'][3]
        self.lookup_move = get_movement_lookup(game_state['self'][3])

    # Rotate Map
    game_state = rotate_map(game_state, self.orientation)

    current_orientation = game_state['self'][3]
    # Check if orientation correct
    self.logger.debug(f'The orientation should be (1,1) and is {current_orientation}')

    random_number = random.uniform(0,1)
    if random_number > epsilon:
        action = np.argmax(qtable[state])
    else:
        # choose random action
        self.action = np.random.choice(self.actions)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    x_self, y_self = game_state['self'][3]
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosion_map']
    field = game_state['field']
    others = game_state['others']

    view = []
    countdowns = []
    directions = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
    for direct in directions:
        x_dir, y_dir = direct
        x, y = x_self + x_dir, y_self + y_dir
        tile_info = field[x,y]
        if x_dir < 0 or y_dir < 0 or x_dir >= len(field) or y_dir >= len(field[0]):
                # Out of bounds, consider it a wall
                view.append(-2)
        elif tile_info == 0:
            for bomb in bombs:
                if bomb[0][0] == x and bomb[0][1] == y: view.append(-2)
                # explosion
            for coin in coins:
                if coin[0] == x and coin[1] == y: view.append(2)
        else:
            view.append(tile_info)

        # Check if the tile is within the bounds of the explosion map.
        if 0 <= x < explosion_map.shape[0] and 0 <= y < explosion_map.shape[1]:
            countdown = explosion_map[x][y]
            normalized_countdown = countdown / 3.0  # Normalize by dividing by the maximum countdown (3)
            countdowns.append(normalized_countdown)
        else:
            countdowns.append(0)  # If out of bounds, countdown is 0.

    # Manhatten Distance to nearest coin
    nearest_coin = float('inf')
    for coin_pos in coins:
        distance = abs(x_self - coin_pos[0]) + abs(y_self - coin_pos[1])
        if distance < nearest_coin:
            nearest_coin = distance

    # Manhatten Distance to nearest bomb
    nearest_bomb = float('inf')
    for bomb_pos in bombs:
        distance = abs(x_self - bomb_pos[0]) + abs(y_self - bomb_pos[1])
        if distance < nearest_bomb:
            nearest_bomb = distance

    # Manhatten Distance to nearest opponent
    nearest_opponent = float('inf')
    for other in others:
        distance = abs(x_self - other[3][0]) + abs(y_self - other[3][1])
        if distance < nearest_opponent:
            nearest_opponent = distance

    return view + countdowns + nearest_coin + nearest_bomb + nearest_opponent
        

    


