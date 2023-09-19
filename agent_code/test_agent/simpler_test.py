import gym
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense


env = gym.make('FrozenLake-v1')


discount_factor = 0.95
eps = 0.5
eps_decay_factor = 0.999
num_episodes = 500


model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(20, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

for i in range(num_episodes):
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    while not done:
        if np.random.random() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(model.predict(np.identity(env.observation_space.n)[state:state + 1]))
        new_state, reward, done, _, _ = env.step(action)
        target = reward + discount_factor * np.max(model.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))
        target_vector = model.predict(np.identity(env.observation_space.n)[state:state + 1])[0]
        target_vector[action] = target
        model.fit(np.identity(env.observation_space.n)[state], target_vector.reshape(-1, env.action_space.n), epochs=1, verbose=0)
        state = new_state