from collections import namedtuple, deque, defaultdict
import pickle
import numpy as np
from typing import List
import random

import events as e
from .callbacks import game_state_to_features  # Add your implementation of rotate_map if needed

# Define Hyperparameters and Constants
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3
BOMB = -10

WINDOW_LENGTH = 7
INPUT_SHAPE = (4, 7)

GAMMA = 0.99
EPSILON = 0.9

# Transition namedtuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters for Q-learning
LEARNING_RATE = 0.1  # Adjust as needed

# Initialize Q-table
q_table = defaultdict(lambda: defaultdict(float))

# Experience replay buffer
TRANSITION_HISTORY_SIZE = 10
experience_replay = deque(maxlen=TRANSITION_HISTORY_SIZE)

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma = GAMMA
    self.epsilon = EPSILON
    self.state_dim = INPUT_SHAPE[0] * INPUT_SHAPE[1]

def store_statistics(self, statistic_dict: dict):
    """
    This function exports statistics about every learning episode.
    Using this, it is possible to create cool plots later on.
    """
    # Example: Store custom statistics relevant to your goals
    statistic_dict["custom_statistic"] = self.custom_statistic


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    This function is used to implement classic Q-learning, where the Q-table is updated based on events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Decrease exploration linearly
    if old_game_state['round'] % 100 == 0 and self.epsilon > 0.1:
        self.epsilon -= 0.1

    # Get old_state and new_state
    old_state = tuple(game_state_to_features(old_game_state))
    new_state = tuple(game_state_to_features(new_game_state))

    # Compute rewards
    reward = reward_from_events(events)

    # Update Q-table based on Q-learning formula
    max_q_value_new_state = max(q_table[new_state].values(), default=0.0)
    q_value_old_state_action = q_table[old_state][self_action]
    q_table[old_state][self_action] = q_value_old_state_action + \
                                      LEARNING_RATE * (reward + GAMMA * max_q_value_new_state - q_value_old_state_action)

def reward_from_events(events: List[str]) -> int:
    """
    Calculate the cumulative reward based on a list of events.

    This function calculates rewards similar to the original `train.py`.

    :param events: A list of events that occurred during the agent's step.
    :return: The cumulative reward based on the events.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 20,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: 0,
        e.BOMB_DROPPED: 1,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 3,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -30,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 20,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: The state in the final step of the game.
    :param last_action: The last action taken by the agent.
    :param events: The events that occurred during the final step.
    """
    # Update the Q-table based on final events and transitions
    old_state = tuple(game_state_to_features(last_game_state))
    reward = self.reward_from_events(events)

    # Update the Q-value for the last action in the final state
    q_value_last_state_action = self.q_table[old_state][last_action]
    self.q_table[old_state][last_action] = q_value_last_state_action + \
        self.alpha * (reward - q_value_last_state_action)

    # Store the final Q-table for future use
    self.save_q_table("q_table.pickle")
