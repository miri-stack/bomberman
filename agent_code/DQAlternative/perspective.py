from collections import deque
import numpy as np
from . import hyperparameter as hp

# Create input values for neural network by looking in every possible direction
def state_to_features(field,game_state):
    #Returns a 4 x distance matrix containing the events in the line of sight in every direction until distance 

    result = np.zeros((4, 6))
    x_self, y_self = game_state['self'][3]

    cols = len(field[0])
    rows = len(field)
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosion_map']
    others = game_state['others']

    for k, pos in enumerate([(-1,0),(1,0),(0,1),(0,-1)]):
        x_pos, y_pos = pos

        # If wall is in this direction fill whole entry with wall
        if field[x_self + x_pos][y_self + y_pos] == -1:
            result[k] = np.full_like(result[k], hp.WALL)
            continue

        # Perform BFS from next field in this direction
        # Initialize the queue with the player's position
        queue = deque([(x_self + x_pos, y_self + y_pos, 0)])

        # Initialize the distance matrix
        distance_matrix = np.full((17, 17), np.inf)
        distance_matrix[x_self + x_pos, y_self + y_pos] = 0

        # Define the possible moves (up, down, right, left)
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Perform BFS
        while queue:
            x, y, steps = queue.popleft()
            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 17 and 0 <= new_y < 17:
                    if field[new_x][new_y] != -1 and steps + 1 < distance_matrix[new_x][new_y]:
                        distance_matrix[new_x][new_y] = steps + 1
                        queue.append((new_x, new_y, steps + 1))

        # Distance to next bomb in this direction
        bombdist = -1
        for bomb in bombs:
            if distance_matrix[bomb[0][0],bomb[0][1]] < bombdist or bombdist == -1:
                bombdist = distance_matrix[bomb[0][0],bomb[0][1]]

        result[k,1] = bombdist

        # Countdown until next field in this direction is dangerous
        if explosions[x_self + x_pos][y_self + y_pos] == 0 and bombdist > 3: # No current explosion and not in radius of unexploded bomb
            badtimer = -1
        elif explosions[x_self + x_pos][y_self + y_pos] > 0: # Current explosion 
            badtimer = 0
        else: 
            badtimer = -1
            for j in range(4):
                for bomb in bombs:
                    if bomb[0][0] == x_self + (j+1)*x_pos and bomb[0][1] == y_self + (j+1)*y_pos: # Look if bomb is in straight line from agent and therefore dangerous
                        if badtimer == -1 or bomb[1] < badtimer:
                            badtimer = bomb[1]

        result[k,2] = badtimer



        # Distance to next coin in this direction
        coindist = -1
        for coin in coins:
            if distance_matrix[coin[0], coin[1]] < coindist or coindist == -1:
                coindist = distance_matrix[coin[0], coin[1]]
        result[k,3] = coindist


        # Distance to next crate in this direction
        indices = np.where(np.array(field) == 1)
        index_pairs = list(zip(indices[0], indices[1]))
        cratedist = -1
        for crate in index_pairs:
            if distance_matrix[crate[0], crate[1]] < cratedist or cratedist == -1:
                cratedist = distance_matrix[crate[0], crate[1]]

        result[k,4] = cratedist


        # Distance to next enemy in this direction
        enemydist = -1
        for o in others:
            if distance_matrix[o[3][0], o[3][1]] < enemydist or enemydist == -1:
                enemydist = distance_matrix[o[3][0], o[3][1]]


        result[k,5] = enemydist

    # Flatten matrix
    result = result.reshape(-1)


    # Countdown until current field becomes dangerous
    # No need to check for current explosion. Agent would be already dead!
    self_countdown = -1

    for j in range(4):
        for bomb in bombs:
            if bomb[0][0] == x_self + j*x_pos and bomb[0][1] == y_self + j*y_pos: # Look if bomb is in straight line from agent and therefore dangerous
                if self_countdown == -1 or bomb[1] < self_countdown:
                    self_countdown = bomb[1]

    # Boolean if bomb is droppable
    bomb_value = np.float64(game_state['self'][2])

    append_part = [[self_countdown, bomb_value]]

    result = np.append(result, append_part)
    result = result.reshape(1,26)

    return result
