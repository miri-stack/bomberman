from collections import defaultdict
import pickle

# Define Hyperparameters and Constants
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3
BOMB = -10

WINDOW_LENGTH = 7
INPUT_SHAPE = (7, 7)

class SimpleQLearningAgent:
    GAMMA = 0.9
    EPSILON = 0.1
    ALPHA = 0.1

    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state = None
        self.action = None
        self.epsilon = self.EPSILON
        self.alpha = self.ALPHA
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]

    def update_q_table(self, old_state, new_state, action, reward):
        max_q_value_new_state = max(self.q_table.get(new_state, {}).values(), default=0.0)
        q_value_old_state_action = self.q_table[old_state].get(action, 0.0)
        self.q_table[old_state][action] = q_value_old_state_action + \
                                               self.alpha * (reward + self.GAMMA * max_q_value_new_state - q_value_old_state_action)

    def save_q_table(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(dict(self.q_table), file)

    def load_q_table(self, file_name):
        with open(file_name, 'rb') as file:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(file))

    def get_q_values(self, state, action):
        state_q_values = self.q_table.get(state, {})
        return state_q_values.get(action, 0.0)