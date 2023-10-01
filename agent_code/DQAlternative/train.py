from collections import namedtuple, deque, defaultdict

import pickle
import numpy as np
from typing import List
import random

import events as e

import tensorflow as tf

from . import hyperparameter as hp
from .orientation import rotate_map, rotate_other_values
from .perspective import state_to_features
from .add_rewards import get_bomb_distance, get_deadly_distance, get_next_to_crate

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 500  # keep only ... last transitions

# Events
DISTANCE_TO_BOMB_INCREASED = "BOMBDISTANCE_INCREASED"
BAD_FIELD_OLD = "BAD_FIELD_OLD"
BAD_FIELD_NEW = "BAD_FIELD_NEW"
NEXT_TO_CRATE = "NEXT_TO_CRATE"
STAYED_ON_FIELD = "STAYED_ON_FIELD"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.statistic_dict = defaultdict(int)
    self.state_array = []
    self.action_array = []
    self.next_state_array = []
    self.reward_array = []
    self.gamma = hp.GAMMA
    self.epsilon = hp.EPSILON
    self.learning_rate = hp.LEARNING_RATE
    self.optimizer = tf.keras.optimizers.Adam(hp.LEARNING_RATE)

    self.model.compile(
    optimizer='adam',     # Choose an optimizer (e.g., 'adam', 'sgd', etc.)
    loss='mean_squared_error',  # Specify the loss function
    metrics=['accuracy']   # Optional: Specify evaluation metrics
    )

    # Setup network
    model_weights_path = "models/q_network_weights.h5"
    # If train and not retrain -> Create new network
    if hp.RETRAIN:
        # Load weights
        self.model.load_weights(model_weights_path)
        

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Decrease exploration linearly
    if old_game_state['round'] % 1000 == 0 and old_game_state['step'] == 1 and self.epsilon > 0.1:
        self.epsilon -= 0.05

    # Idea: Add your own events to hand out rewards


    # Reward dropping a bomb next to a crate
    if get_next_to_crate(old_game_state, self_action):
        events.append(NEXT_TO_CRATE)

    # Reward increasing distance to bombs
    old_bomb_dist = get_bomb_distance(old_game_state)
    new_bomb_dist = get_bomb_distance(new_game_state)

    if new_bomb_dist > old_bomb_dist:
        events.append(DISTANCE_TO_BOMB_INCREASED)
        

    # Get old_game_state
    self.field = rotate_map(old_game_state, self.orientation)
    old_game_state_adjusted = rotate_other_values(old_game_state, self.orientation)
    old_state = state_to_features(self.field, old_game_state_adjusted)
   
    # Get new_game_state
    self.field = rotate_map(new_game_state, self.orientation)
    new_game_state_adjusted = rotate_other_values(new_game_state, self.orientation)
    new_state= state_to_features(self.field, new_game_state_adjusted)

    # Penalize when new field is old field
    if old_game_state['self'][3] == new_game_state['self'][3]:
        events.append(STAYED_ON_FIELD)
    
    # Compute rewards
    rewards = reward_from_events(self, events)

    # Penalize when agent was on dangerous field in old state
    if old_state[0,25] > 0:
        rewards -= 10*old_state[0,25]


    # Penalize when agent was on dangerous field in new state
    if new_state[0,25] > 0:
        rewards -= 10*new_state[0,25]

    #self.logger.info(f"Awarded {rewards} for events {', '.join(events)}")
        

    self.state_array.append(old_state)
    self.action_array.append(self.next_action_adj)
    self.next_state_array.append(new_state)
    self.reward_array.append(rewards)


    # Train the model
    if len(self.state_array) >= TRANSITION_HISTORY_SIZE and self.stepcount % 500 == 0:  # Batch size for training
        self.stepcount = 0

        old_states = np.vstack(self.state_array)
        actions = self.action_array
        rewards = np.array(self.reward_array)
        next_states = np.vstack(self.next_state_array)

        # Compute the Q-values for the current and next states
        q_values = self.model.predict(old_states)
        next_q_values = self.model.predict(next_states)

        # Compute the target Q-values
        targets = q_values.copy()
        for i in range(TRANSITION_HISTORY_SIZE):
            action_index = self.actions.index(actions[i])
            targets[i, action_index] = q_values[i, action_index] + self.learning_rate*(rewards[i] + self.gamma * np.max(next_q_values[i, action_index]) - q_values[i, action_index])

        # Train the model
        self.model.fit(old_states, targets, epochs=1, verbose=0)
        self.state_array = []
        self.action_array = []
        self.next_state_array = []
        self.reward_array = []


    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.state_array.append(state_to_features(self.field, last_game_state))
    self.action_array.append(last_action)
    self.next_state_array.append(state_to_features(self.field, last_game_state))
    self.reward_array.append(reward_from_events(self, events))

    # Store the model in every 10th episode
    if last_game_state['round'] % 5000 == 0:
        model_weights_path = "models/model_weights" + str(last_game_state['round']) + ".h5"
        self.model.save_weights(model_weights_path)

    # Store the statistics
    if last_game_state['round'] % 1000 == 0:
        with open("stats/statistics_" + str(last_game_state['round']) + ".pt", "wb") as stat_file:
            pickle.dump(self.statistic_dict, stat_file)

        # Initialise new storage dict
        self.statistic_dict = defaultdict(int)

    


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT:2,
        e.MOVED_RIGHT:2,
        e.MOVED_UP:2,
        e.MOVED_DOWN:2,
        e.BOMB_DROPPED:10,
        e.CRATE_DESTROYED:20,
        e.WAITED:-1,
        e.INVALID_ACTION:-100,
        e.COIN_COLLECTED: 50,
        e.KILLED_SELF:-1000,
        e.GOT_KILLED:-100000,
        e.SURVIVED_ROUND: 1000,
        e.KILLED_OPPONENT: 80,
        NEXT_TO_CRATE: 30,
        STAYED_ON_FIELD: -10,
        DISTANCE_TO_BOMB_INCREASED: 100
    }


    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            self.statistic_dict[str(event)] += 1 # Update statistics
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    self.statistic_dict["reward"] += reward_sum
    return reward_sum
