import events as e
import pickle, os

# define parameters
# alpha, gamma epsilon
ALPHA = 0.001
GAMMA = 0.99
EPSILON = 0.9

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# update action value funtion
# action selection: (epsilon) greedy for exploration decrease after n episodes 

def setup_training(self):
''' called when loading the agent, after calling setup in callbacks.py.
Use this to initialize variables you only need for training.
'''
  self.statistic_dict = None
  self.gamma = self.GAMMA
  self.epsilon = self.EPSILON

  # self.state_dim = hp.INPUT_SHAPE[0]*hp.INPUT_SHAPE[1]
  self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
  # self.qtable = pd.DataFrame(columns=ACTIONS,dtype=np.float64)


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

    # Get old_game_state
    old_game_state_adjusted = rotate_map(old_game_state, self.orientation)
    old_state = state_to_features(old_game_state_adjusted)
  
    # Get new_game_state
    new_game_state_adjusted = rotate_map(new_game_state, self.orientation)
    new_state = state_to_features(new_game_state_adjusted)

    reward = get_reward_from_events(events)
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward))

    last_transition = transitions[-1]

    if last_transition.state not in self.qtable:
      self.qtable[last_transition.state] = {}
    if not self.qtable[old_game_state][action]:
       self.qtable[old_game_state]
    self.qtable[last_transition.state][last_transition.action] = qtable[last_transition.state, last_transition.action] + self.alpha * (last_transition.reward + self.gamma * (self.qtable[old_game_state,self_action] - self.qtable[last_transition.state,last_transition.action])


    

def end_of_round(self, last_game_state, last_action, events):
'''is very similar to the previous, but only called once per agent after the last step of a round '''
  # Store the model in every 10th episode
  if last_game_state['round'] % 10 == 0:
      with open("save_files/table_" + str(last_game_state['round']) + ".pt","w") as file:
            pickle.dump(self.Q, file)




def get_reward_from_events(self, events) -> int:
        game_rewards = {
            e.COIN_COLLECTED: 100,
            e.KILLED_OPPONENT: 500,
            e.MOVED_RIGHT: 1,
            e.MOVED_LEFT: 1,
            e.MOVED_UP: 1,
            e.MOVED_DOWN: 1,
            e.WAITED: -1,
            e.INVALID_ACTION: -200,
            e.BOMB_EXPLODED: 50,
            e.CRATE_DESTROYED: 100,
            e.COIN_FOUND: 100,
            e.BOMB_DROPPED: 10,
            e.KILLED_SELF: -500,
            e.GOT_KILLED: -700,
        }
        reward_sum = 0
        for event in events:
             reward_sum += game_rewards[event]
        self.logger("Rewards granted for this round")
        # table or something for those
      return reward_sum

