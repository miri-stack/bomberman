import numpy as np
import pickle
import events as e


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    self.logger.info('Mary Shelly is getting ready.')
    if self.train:
        
    else:
        with open(qtable_load, "rb") as file:
            self.qtable = pickle.load(file)
    # initialize Q(S,A) to 0 for all S and A
    # Start Field 


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    # observe environment

    
    # return game_state['user_input']

