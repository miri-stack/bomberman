import events as e

# define parameters
# alpha, gamma epsilon

# update action value funtion
# action selection: (epsilon) greedy for exploration decrease after n episodes 

def setup_training(self):
''' called when loading the agent, after calling setup in callbacks.py.
Use this to initialize variables you only need for training.
'''
    self.qtable = pd.DataFrame(columns=ACTIONS,dtype=np.float64)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
''' is called once after each step except the last. At this point, all the actions have been executed and their consequences are known. Use this callback to collect training data and ll an
experience buffer '''
    reward = get_reward_from_events(events)
    qtable[old_game_state, self_action] = qtable[old_game_state, self_action] + self.alpha * (reward + self.gamma * (qtable[new_game_state,self_action] - qtable[old_game_state,self_action])


def end_of_round(self, last_game_state, last_action, events):
'''is very similar to the previous, but only called once per agent after the last step of a round '''

def epsilon_greedy_policy(self, qtable, state, epsilon):
  random_number = random.uniform(0,1)
  if random_number > epsilon:
    action = np.argmax(qtable[state])
  else:
    # choose random action
    action = 
  return action

def get_reward_from_events(self, events) -> int:
        game_rewards = {
            e.COIN_COLLECTED: 100,
            e.KILLED_OPPONENT: 500,
            e.MOVED_RIGHT: 1,
            e.MOVED_LEFT: 1,
            e.MOVED_UP: 1,
            e.MOVED_DOWN: 1,
            e.WAITED: -1,
            e.INVALID_ACTION: -10,
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

