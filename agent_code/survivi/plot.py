import matplotlib.pyplot as plt
import os
import pickle

with open('agent_code/coini/concatenated_data.pt', 'rb') as file:
        existing_data = pickle.load(file)

for key in existing_data.keys():
        plt.figure()
        plt.plot(existing_data[key], label='coini')
        plt.title(str(key))
        plt.savefig('agent_code/coini/plots/'+str(key)+'.png')

