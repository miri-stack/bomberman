import numpy as np
import pickle, os
import events as e
import copy, random
import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
WALL = 0
FREE = 1
CRATE = 2
COIN = 5
PLAYER = 0
BOMB = -5
OTHER = 0

WINDOW_LENGTH = 11
INPUT_SHAPE = (WINDOW_LENGTH, WINDOW_LENGTH)

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
    self.actions = ACTIONS

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
    self.logger.info('Pick action')
    # observe environment

    if game_state['step'] == 1: # First round, setup orientation
        self.orientation = game_state['self'][3]
        self.lookup_move = get_movement_lookup(game_state['self'][3])

    # Rotate Map
    game_state = rotate_map(game_state, self.orientation)
    

    current_orientation = game_state['self'][3]
    # Check if orientation correct
    if game_state['step'] == 1:
        self.logger.debug(f'The orientation should be (1,1) and is {current_orientation}')
    state = state_to_features(game_state)

    if self.train:

        random_number = random.uniform(0,1)
        # choose random action
        if random_number < self.epsilon:
            action = np.random.choice(self.actions)
            return action
    if tuple(state) not in self.qtable:
        action=  np.random.choice(ACTIONS,p=[.2, .2, .2, .2, .1, .1])
    else:
        action = np.argmax(self.qtable[tuple(state)])
    return action


def state_to_features(game_state: dict):
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
    explosions = np.array(game_state['explosion_map'])
    field = np.array(game_state['field'])
    others = game_state['others']
    rows, cols = field.shape[0], field.shape[1]
    observation = np.zeros([rows, cols], dtype=np.float32)

    # Taken from Bomb Class in items.py
    def get_blast_coords(x, y, arena):
        blast_coords = [(x, y)]

        for i in range(1, s.BOMB_POWER + 1):
            if arena[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, s.BOMB_POWER + 1):
            if arena[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, s.BOMB_POWER + 1):
            if arena[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, s.BOMB_POWER + 1):
            if arena[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

        return blast_coords
    
    if bombs:
        for bomb in bombs:
            (x,y), timer = bomb
            blast_coord = get_blast_coords(x,y,field)
            for coord in blast_coord:
                observation[coord] = timer + BOMB
                if coord == (x_self,y_self):
                    observation[coord] = BOMB*10
    observation[np.where(explosions != 0)[0],np.where(explosions != 0)[1]] = BOMB
    
        
    if coins:
        for co in coins:
            observation[co[0], co[1]] = COIN
    
    if others:
        for ot in others:
            observation[ot[3][0],ot[3][1]] = OTHER


    halfdistance = int(np.floor(WINDOW_LENGTH/2))
    padded_observation = np.pad(observation, halfdistance, constant_values=WALL)

    # Calculate the slice for the view within the padded observation matrix
    x_slice = slice(int(x_self), int(x_self) + 2 * halfdistance + 1)
    y_slice = slice(y_self, y_self + 2 * halfdistance + 1)
    # Create the view by extracting the subarray from the padded observation matrix
    view = padded_observation[x_slice, y_slice]

    return view.reshape(-1)
        
    


