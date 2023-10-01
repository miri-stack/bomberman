import matplotlib.pyplot as plt
import os
import pickle

# Load new statistics
with open('agent_code/BombiMcBombFace/concatenated_data.pt', 'rb') as file:
        existing_data = pickle.load(file)

with open('agent_code/BombiMcBombFace/concatenated_data2.pt', 'rb') as file:
        existing_data2 = pickle.load(file)

# Load old statistics with bugs
with open('agent_code/BombiMcBombFace/concatenated_data2.pt', 'rb') as file:
        bug_data = pickle.load(file)

for key in existing_data.keys():
        plt.figure()
        plt.plot(existing_data[key] + existing_data2[key], label='corrected version')
        plt.plot(bug_data[key], label='original version')
        plt.title('DQN – ' + str(key))
        plt.xlabel("[1000 rounds]")
        plt.savefig('agent_code/BombiMcBombFace/plots/'+str(key)+'.png')

