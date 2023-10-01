import pickle

# Load the Q-table from the pickle file
with open("./agent_code/qagent/q_table.pickle", "rb") as file:
    q_table = pickle.load(file)

# Print the Q-table
for state, action_values in q_table.items():
    print(f"State: {state}")
    for action, q_value in action_values.items():
        print(f"  Action: {action}, Q-Value: {q_value}")
