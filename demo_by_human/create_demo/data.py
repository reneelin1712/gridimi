import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

states = np.load('../data/states.npy', allow_pickle=True)
actions = np.load('../data/actions.npy', allow_pickle=True)
print('states',states)
reshaped_states = states.reshape(states.shape[0], states.shape[1], 1)
reshaped_actions = actions.reshape(actions.shape[0], actions.shape[1], 1)
print('reshaped_states',reshaped_states)
    
demon_list = []

for state_traj, action_traj in zip(reshaped_states, reshaped_actions):
    state_traj = np.array(state_traj)
    action_traj = np.array(action_traj)
    
    # Ensure that state_traj is one step longer than action_traj
    assert state_traj.shape[0] == action_traj.shape[0] + 1
    
    # Append action 0 to the end of action_traj
    action_traj = np.append(action_traj, 0)

    # link_ids = state_traj[:, :-1].astype(int) 
    link_ids = state_traj[:, :1]
    
    if None in link_ids:
        print("Warning: Some (u, v) pairs were not found in link_mapping.")
    
    # Use link_ids instead of node pairs for further computations
    state_traj = link_ids
    print('state_traj',state_traj)
    
    # Reshape action_traj to make it a column vector
    action_traj = action_traj.reshape(-1, 1)

    # Concatenate state_traj and action_traj along the second axis
    concated_array = np.hstack((state_traj, action_traj))

    demon_list.append(concated_array)

with open("../data/demon.pkl", "wb") as f:
    pickle.dump(demon_list, f)

# valid_trajectories = [traj for traj in demon_list if all(np.isin(traj[:, -1], [0, 1, 2, 3]))]

train_list, test_list = train_test_split(demon_list, test_size=0.2, random_state=42)

with open("../data/demon_train.pkl", "wb") as f:
    pickle.dump(train_list, f)

with open("../data/demon_test.pkl", "wb") as f:
    pickle.dump(test_list, f)

print('demon_list',demon_list)
print('demon_list len',len(demon_list))







