from collections import namedtuple, deque, defaultdict

import pickle
import numpy as np
from typing import List
import random

import events as e
from .callbacks import state_to_features, rotate_map

import tensorflow as tf

# Define Hyperparameter
WALL = -1
EXPLOSION = -3
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3
BOMB = -10

WINDOW_LENGTH = 7
INPUT_SHAPE = (4, 7)

LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 0.9

RETRAIN = False

def store_statistics(statistic_dict: dict):
    """ This function exports statistics about every learning episode.
    Using this, it is possible to create cool plots later on. """

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.statistic_dict = None
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma = GAMMA
    self.epsilon = EPSILON
    self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    self.state_dim = INPUT_SHAPE[0] * INPUT_SHAPE[1]

    # Setup network
    model_weights_path = "q_network_weights.h5f"
    # If train and not retrain -> Create a new network
    if RETRAIN:
        # Load weights
        self.model.load_weights(model_weights_path)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
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
        PLACEHOLDER_EVENT: -0.1  # idea: the custom event is bad
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            self.statistic_dict[str(event)] += 1  # Update statistics
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    self.statistic_dict["reward"] += reward_sum
    return reward_sum


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
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Setup statistics dictionary if necessary
    if old_game_state['step'] == 1:  # Initialise a new dict
        self.statistic_dict = defaultdict(int)  # defaultdict with 0 as the default value

    # Decrease exploration linearly
    if old_game_state['round'] % 100 == 0 and self.epsilon > 0.1:
        self.epsilon -= 0.1

    # Idea: Add your own events to hand out rewards
    # TODO: Add "more distance to bombs" event
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # Get old_game_state
    old_game_state_adjusted = rotate_map(old_game_state, self.orientation)
    old_state = state_to_features(old_game_state_adjusted, distance=WINDOW_LENGTH).reshape(1, self.input_dim)

    # Get new_game_state
    new_game_state_adjusted = rotate_map(new_game_state, self.orientation)
    new_state = state_to_features(new_game_state_adjusted, distance=WINDOW_LENGTH).reshape(1, self.input_dim)

    # Compute rewards
    rewards = reward_from_events(self, events)

    self.transitions.append(Transition(old_state, self.next_action_adj, new_state, rewards))

    # Learn something
    with tf.GradientTape() as tape:
        current_transition = random.choice(self.transitions)
        q_values = self.model(current_transition.state)
        q_value = q_values[0][self.actions.index(self.next_action_adj)]
        target = current_transition.reward + self.gamma * tf.reduce_max(self.model(current_transition.next_state))

        loss = tf.keras.losses.mean_squared_error(np.array([target]), np.array([q_value]))

        grads = tape.gradient(loss, self.model.trainable_variables)
        print(f'grads shape {len(grads)}')
        print(grads)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in the final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    model_weights_path = "models/model_weights" + str(last_game_state['round']) + ".h5"
    self.model.save_weights(model_weights_path)

    # Store the statistics
    with open("stats/statistics_" + str(last_game_state['round']) + ".pt", "wb") as stat_file:
        pickle.dump(self.statistic_dict, stat_file)
