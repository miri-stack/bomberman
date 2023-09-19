from collections import deque
import numpy as np
from . import hyperparameter as hp

def get_bomb_distance(matrix, bomb_list):
    """ Input: The perspective matrix of the agent 
        Returns: The sum of the bomb distances in the perspective weighted by the 
                bomb countdown
    """
    
    factor = 1 # Define the weighting parameter to give the countdown more weight if necessary
    bombsum = 0 # Initialize the sum

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
    for bomb in bomb_list:
        bombsum += distance_matrix[bomb[0][0],bomb[0][1]] * bomb[1] * factor
    

    return bombsum


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
