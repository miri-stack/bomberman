from collections import deque
import numpy as np
from . import hyperparameter as hp

def get_next_to_crate(old_game_state, action):
    if not action == "BOMB":
        return False
    
    else:
        pos = old_game_state['self'][3]
        for move in [(0,1),(1,0),(-1,0),(0,-1)]:
            for i in range(1,4):
                if 0 <= pos[0]+i*move[0] < 17 and 0 <= pos[1]+i*move[1] < 17:
                    if old_game_state['field'][pos[0]+i*move[0]][pos[1]+i*move[1]] == -1:
                        break
                    elif old_game_state['field'][pos[0]+i*move[0]][pos[1]+i*move[1]] == 1:
                        return True

    return False


def get_bomb_distance(game_state):
    """ Input: The perspective matrix of the agent 
        Returns: The sum of the bomb distances in the perspective weighted by the 
                bomb countdown
    """

    bombsum = 0 # Initialize the sum
    bombs = game_state['bombs']
    if len(bombs) == 0:
        return np.inf

    self_pos = game_state['self'][3]

    map = game_state['field']

    # Initialize the queue with the player's position
    queue = deque([(self_pos[0], self_pos[1], 0)])

    # Initialize the distance matrix
    distance_matrix = np.full((17, 17), np.inf)
    distance_matrix[self_pos[0], self_pos[1]] = 0

    # Define the possible moves (up, down, right, left)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Perform BFS
    while queue:
        x, y, steps = queue.popleft()
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 17 and 0 <= new_y < 17:
                if self_pos[0]-3 <= new_x <= self_pos[0]+3 and self_pos[1]-3 <= new_y <= self_pos[1]+3:
                    if map[new_x, new_y] != -1 and steps + 1 < distance_matrix[new_x, new_y]:
                        distance_matrix[new_x, new_y] = steps + 1
                        queue.append((new_x, new_y, steps + 1))

    # Calculate the distance to bombs
    bomb_count = 0
    for bomb in bombs:
        if not distance_matrix[bomb[0][0],bomb[0][1]] == np.inf:
            bombsum += distance_matrix[bomb[0][0],bomb[0][1]]**3 * bomb[1]
            bomb_count += 1

    if bomb_count == 0:
        return np.inf
        
    return bombsum/bomb_count


def get_deadly_distance(matrix):
    """ Input: The perspective matrix of the agent 
        Returns: The sum of the distances to deadly fields in the perspective """

    deadly_sum = 0 # Initialize the sum

    # Initialize the queue with the player's position
    queue = deque([(hp.WINDOW_LENGTH // 2, hp.WINDOW_LENGTH // 2, 0)])

    # Initialize the distance matrix
    distance_matrix = np.full((hp.WINDOW_LENGTH, hp.WINDOW_LENGTH), np.inf)
    distance_matrix[hp.WINDOW_LENGTH // 2, hp.WINDOW_LENGTH // 2] = 0

    # Define the possible moves (up, down, right, left)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Perform BFS
    while queue:
        x, y, steps = queue.popleft()
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < hp.WINDOW_LENGTH and 0 <= new_y < hp.WINDOW_LENGTH:
                if matrix[new_x, new_y] != hp.WALL and matrix[new_x, new_y] != hp.CRATE and steps + 1 < distance_matrix[new_x, new_y]:
                    distance_matrix[new_x, new_y] = steps + 1
                    queue.append((new_x, new_y, steps + 1))

    # Calculate the distance to bombs
    for x in range(hp.WINDOW_LENGTH):
        for y in range(hp.WINDOW_LENGTH):
            if matrix[x, y] == hp.BOMB or matrix[x,y] == hp.EXPLOSION:
                deadly_sum += distance_matrix[x,y]   

    return deadly_sum
