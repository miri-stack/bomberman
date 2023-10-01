from collections import namedtuple, deque
import json
import os
from typing import List

import events as e
from .callbacks import game_state_to_features
from .qmodel import SimpleQLearningAgent

# TODO: input should be the same but change rewards
# training phases: step by step rewards anpassen, sodass spezifische sachen trainiert werden (wie coins sammeln, bombs vermeiden)
# retrain with new rewards and previous knowledge
# --agent you have to specify all agents that are to be used
# different scenarios in settings.py 

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

# Hyperparameters
GAMMA = 0.95
EPSILON = 0.3
ALPHA = 0.3

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Transition namedtuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters for Q-learning
LEARNING_RATE = 0.001  # Adjust as needed

# Experience replay buffer
TRANSITION_HISTORY_SIZE = 100
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
    self.alpha = 0.1

    # Check if the Q-table file exists
    if os.path.isfile("q_table.pickle"):
        self.q_table = SimpleQLearningAgent()
        self.q_table.load_q_table("q_table.pickle")
    else:
        # Initialize Q-table here as an instance of SimpleQLearningAgent
        self.q_table = SimpleQLearningAgent()  # Corrected initialization

def store_statistics(self, statistic_dict: dict):
    """
    This function exports statistics about every learning episode.
    Using this, it is possible to create cool plots later on.
    
    :param statistic_dict: A dictionary to store statistics.
    """
    # Store custom statistics relevant to your goals
    statistic_dict["custom_statistic"] = self.custom_statistic

    # Store some common statistics
    statistic_dict["epsilon"] = self.epsilon  # Epsilon value for exploration
    statistic_dict["round"] = self.current_round  # Current round number
    statistic_dict["step"] = self.current_step  # Current step number within the round

    # You can add more statistics as needed

    # Example: Store Q-table size
    statistic_dict["q_table_size"] = len(self.q_table.q_table)

    # Example: Store average reward for the last few episodes
    if len(self.episode_rewards) > 0:
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        statistic_dict["average_reward"] = avg_reward

    # Save the statistics to a file
    with open("training_statistics.json", "a") as file:
        json.dump(statistic_dict, file)
        file.write("\n")



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
    old_state = tuple(str(val) for val in game_state_to_features(old_game_state))
    new_state = tuple(str(val) for val in game_state_to_features(new_game_state))
    # Compute rewards
    reward = reward_from_events(events)

    # Update Q-table based on Q-learning formula
    self.q_table.update_q_table(old_state, new_state, self_action, reward)

def reward_from_events(events: List[str]) -> int:
    """
    Calculate the cumulative reward based on a list of events.

    This function calculates rewards similar to the original `train.py`.

    :param events: A list of events that occurred during the agent's step.
    :return: The cumulative reward based on the events.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 100,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.BOMB_DROPPED: 50,
        e.BOMB_EXPLODED: 20,
        e.CRATE_DESTROYED: 30,
        e.COIN_FOUND: 50,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -200,
        e.OPPONENT_ELIMINATED: 80,
        e.SURVIVED_ROUND: 50,
        e.INVALID_ACTION: -200,
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
    reward = reward_from_events(events)

    # Update the Q-value for the last action in the final state
    self.q_table.update_q_table(old_state, None, last_action, reward)

    # Store the final Q-table for future use
    self.q_table.save_q_table("q_table.pickle")
