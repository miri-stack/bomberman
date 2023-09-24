from collections import namedtuple, deque, defaultdict
import events as e
import pickle, os
from .callbacks import state_to_features, rotate_map
from typing import List

# define parameters
# alpha, gamma epsilon
ALPHA = 0.7
GAMMA = 0.8
EPSILON = 0.4

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# update action value funtion
# action selection: (epsilon) greedy for exploration decrease after n episodes 

def setup_training(self):
  ''' called when loading the agent, after calling setup in callbacks.py.
  Use this to initialize variables you only need for training.
  '''
  self.statistics_dict = None
  self.gamma = GAMMA
  self.epsilon = EPSILON
  self.alpha = ALPHA

  # self.state_dim = hp.INPUT_SHAPE[0]*hp.INPUT_SHAPE[1]
  self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
  # self.qtable = pd.DataFrame(columns=ACTIONS,dtype=np.float64)

def update_qtable (self, old_game_state, self_action, events):
  last_transition = self.transitions[-1]
  q_value_old = self.qtable.get(tuple(last_transition.state), {}).get(last_transition.action, 0.0)
  q_value_next = self.qtable.get(tuple(old_game_state),{}).get(self_action, 0.0)
  q_value_new = q_value_old + self.alpha * (last_transition.reward + self.gamma * q_value_next - q_value_old)
  if tuple(last_transition.state) not in self.qtable:
    self.qtable[tuple(last_transition.state)] = {}
  self.qtable[tuple(last_transition.state)][last_transition.action] = q_value_new
  """
  if e.KILLED_SELF or e.GOT_KILLED in events:
    reward = get_reward_from_events(self,events)
    if tuple(old_game_state) not in self.qtable:
      self.qtable[tuple(old_game_state)] = {}
    self.qtable[tuple(old_game_state)][self_action] = q_value_next + self.alpha * (reward - q_value_next)
  """


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
      self.statistics_dict = defaultdict(int) # defaultdict with 0 as default value
      self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # Reset transitions


  # Decrease exploration linearly
  if old_game_state['round'] % 100 == 0 and self.epsilon > 0.1:
      self.epsilon -= 0.1
  # Idea: Add your own events to hand out rewards

  # Get old_game_state
  old_game_state_adjusted = rotate_map(old_game_state, self.orientation)
  old_state = state_to_features(old_game_state_adjusted)

  # Get new_game_state
  new_game_state_adjusted = rotate_map(new_game_state, self.orientation)
  new_state = state_to_features(new_game_state_adjusted)

  reward = get_reward_from_events(self, events)
  self.transitions.append(Transition(old_state, self_action, new_state, reward))
  update_qtable(self, old_state,self_action, events)

  """"
  last_transition = self.transitions[-1]
  q_value_old = self.qtable.get(tuple(last_transition.state), {}).get(last_transition.action, 0.0)
  q_value_next = self.qtable.get(tuple(old_game_state),{}).get(self_action, 0.0)
  q_value_new = q_value_old + self.alpha * (last_transition.reward + self.gamma * (q_value_next - q_value_old))
  if tuple(last_transition.state) not in self.qtable:
    self.qtable[tuple(last_transition.state)] = {}
  self.qtable[tuple(last_transition.state)][last_transition.action] = q_value_new
  """



def end_of_round(self, last_game_state, last_action, events):
  '''is very similar to the previous, but only called once per agent after the last step of a round '''
  if e.SURVIVED_ROUND in events:
        self.logger.info("Round survived!")
  self.statistics_dict['score'] = last_game_state['self'][1]
  # Get old_game_state
  old_game_state_adjusted = rotate_map(last_game_state, self.orientation)
  old_state = state_to_features(old_game_state_adjusted)
  # Store the model in every 100th episode
  update_qtable(self, old_state, last_action, events)
  if last_game_state['round'] % 100 == 0:
    with open("save_files/table_" + str(last_game_state['round']) + ".pt","wb") as file:
      pickle.dump(self.qtable, file)
    with open("save_files/statistics_" + str(last_game_state['round']) + ".pt", "wb") as stat_file:
      pickle.dump(self.statistics_dict, stat_file)




def get_reward_from_events(self, events) -> int:
  game_rewards = {
      e.COIN_COLLECTED: 100,
      e.KILLED_OPPONENT: 500,
      e.MOVED_RIGHT: 10,
      e.MOVED_LEFT: 10,
      e.MOVED_UP: 10,
      e.MOVED_DOWN: 10,
      e.WAITED: -1,
      e.INVALID_ACTION: -200,
      e.BOMB_EXPLODED: 0,
      e.CRATE_DESTROYED: 100,
      e.COIN_FOUND: 100,
      e.BOMB_DROPPED: 0,
      e.KILLED_SELF: -100,
      e.GOT_KILLED: -200,
      e.SURVIVED_ROUND: 500
  }
  reward_sum = 0
  for event in events:
        if event in game_rewards:
          reward_sum += game_rewards[event]
          self.statistics_dict[str(event)] += 1 # Update statistics
  self.logger.info("Rewards granted for this round")
  self.statistics_dict["reward"] += reward_sum
  # table or something for those
  return reward_sum

