import copy

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
    
