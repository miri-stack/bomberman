import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# Load new statistics
with open('agent_code/BombiMcBombFace/concatenated_data.pt', 'rb') as file:
        existing_data = pickle.load(file)

with open('agent_code/BombiMcBombFace/concatenated_data2.pt', 'rb') as file:
        existing_data2 = pickle.load(file)


# Load statistics with alternative features
with open('agent_code/BombiMcBombFace/concatenated_data_alternative.pt', 'rb') as file:
        alternative_data = pickle.load(file)

for key in existing_data.keys():
        plt.figure()
        plt.plot(existing_data[key] + existing_data2[key], label='Close environment corrected', color='orange')
        plt.plot(alternative_data[key], label='Explicit information', color='green')
        plt.title('DQN – ' + str(key))
        if key == "reward":
                print("reward with legend.")
                plt.legend()
        plt.xlabel("Amount of training episodes [1000 rounds]")
        plt.ylabel("Count of events cumulated over 1000 rounds")
        plt.savefig('agent_code/BombiMcBombFace/plots/'+str(key)+'.png')


plt.figure()
colors = ["orange", "green"]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(2)]
labels = ['Close environment corrected', 'Explicit information']
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

def export_legend(legend, filename="agent_code/BombiMcBombFace/plots/legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)

