from collections import namedtuple, deque, defaultdict

import pickle
import numpy as np
from typing import List
import random

import events as e

import tensorflow as tf

from . import hyperparameter as hp
from .orientation import rotate_map
from .perspective import state_to_features
from .add_rewards import get_bomb_distance, get_deadly_distance

# Command for training
# python main.py play --no-gui --agents killi peaceful_agent --train 1 --scenario empty
# State 2
# python main.py play --no-gui --agents killi --train 1 --scenario empty
# State 3
# python main.py play --no-gui --agents killi --train 1 --scenario below_classic
# State 4
# python main.py play --no-gui --agents killi --train 1 --scenario classic


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
CHANGED_BOMB_DISTANCE = "BOMB_DISTANCE"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    print("Setup train is called")
    self.statistic_dict = None
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma = hp.GAMMA
    self.epsilon = hp.EPSILON
    self.optimizer = tf.keras.optimizers.Adam(hp.LEARNING_RATE)
    self.state_dim = hp.INPUT_SHAPE[0]*hp.INPUT_SHAPE[1]
    self.bomb_dist_change = 0

    self.model.compile(
    optimizer='adam',     # Choose an optimizer (e.g., 'adam', 'sgd', etc.)
    loss='mean_squared_error',  # Specify the loss function
    metrics=['accuracy']   # Optional: Specify evaluation metrics
    )

    # Setup network
    model_weights_path = "q_network_weights.h5f"
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
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Setup statistics dictionary if necessary
    if old_game_state['step'] == 1: # Initialise new dict
        self.statistic_dict = defaultdict(int) # defaultdict with 0 as default value
        self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # Reset transitions


    # Decrease exploration linearly
    if old_game_state['round'] % 100 == 0 and self.epsilon > 0.1:
        self.epsilon -= 0.1


    # Idea: Add your own events to hand out rewards
    # TODO: Add "more distance to bombs" event
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # Get old_game_state
    old_game_state_adjusted = rotate_map(old_game_state, self.orientation)
    old_state, old_bomb_list = state_to_features(old_game_state_adjusted)
   
    # Get new_game_state
    new_game_state_adjusted = rotate_map(new_game_state, self.orientation)
    new_state, new_bomb_list = state_to_features(new_game_state_adjusted)
    
    # Add rewards
    old_bomb_dist = get_bomb_distance(old_state, old_bomb_list)
    new_bomb_dist = get_bomb_distance(new_state, new_bomb_list)
    if old_bomb_dist != new_bomb_dist:
        events.append(CHANGED_BOMB_DISTANCE)
        self.bomb_dist_change = new_bomb_dist - old_bomb_dist

    # Compute rewards
    rewards = reward_from_events(self, events)

    # Reshape states
    old_state = old_state.reshape(1,self.input_dim)
    new_state = new_state.reshape(1,self.input_dim)

    self.transitions.append(Transition(old_state, self.next_action_adj, new_state, rewards))

    # Train the model
    if len(self.transitions) >= TRANSITION_HISTORY_SIZE:  # Batch size for training
        transitions = random.sample(self.transitions, TRANSITION_HISTORY_SIZE)

        old_states = np.vstack([t.state for t in transitions])
        actions = [t.action for t in transitions]
        rewards = np.array([t.reward for t in transitions])
        next_states = np.vstack([t.next_state for t in transitions])

        # Compute the Q-values for the current and next states
        q_values = self.model.predict(old_states)
        next_q_values = self.model.predict(next_states)

        # Compute the target Q-values
        targets = q_values.copy()
        for i in range(TRANSITION_HISTORY_SIZE):
            targets[i, self.actions.index(actions[i])] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        history = self.model.fit(old_states, targets, epochs=1, verbose=0)


    


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
    
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state)[0].reshape(1,self.input_dim), last_action, None, reward_from_events(self, events)))

    # Store the model in every 10th episode
    if last_game_state['round'] % 10 == 0:
        model_weights_path = "models/model_weights" + str(last_game_state['round']) + ".h5"
        self.model.save_weights(model_weights_path)

    # Store the statistics
    with open("stats/statistics_" + str(last_game_state['round']) + ".pt", "wb") as stat_file:
        pickle.dump(self.statistic_dict, stat_file)

    


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.KILLED_OPPONENT: 100,
        e.MOVED_LEFT:1,
        e.MOVED_RIGHT:1,
        e.MOVED_UP:1,
        e.MOVED_DOWN:1,
        e.WAITED:0,
        e.BOMB_DROPPED:1,
        e.BOMB_EXPLODED:0,
        e.CRATE_DESTROYED:2,
        e.KILLED_SELF:-100,
        e.OPPONENT_ELIMINATED:0,
        e.SURVIVED_ROUND:20,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            self.statistic_dict[str(event)] += 1 # Update statistics
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    self.statistic_dict["reward"] += reward_sum
    return reward_sum
