import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

states = np.load('data/traj_20181029_dX_0830_0900_states_sameOD.npy', allow_pickle=True)
actions = np.load('data/traj_20181029_dX_0830_0900_actions_sameOD.npy', allow_pickle=True)

demon_list = []
num_dimensions = 427

# Create an instance of the OneHotEncoder
def one_hot_encode_link_ids(link_ids, num_dimensions):
    one_hot_encoded = np.zeros((len(link_ids), num_dimensions), dtype=float)
    for i, link_id in enumerate(link_ids):
        one_hot_encoded[i, link_id] = 1.
    return one_hot_encoded

def one_hot_encode_road_types(road_types):
    # Create a zero array of shape (number of samples, 4)
    one_hot_encoded = np.zeros((len(road_types), 4), dtype=float)
    for i, road_type in enumerate(road_types):
        one_hot_encoded[i, int(road_type)] = 1.
    return one_hot_encoded

for state_traj, action_traj in zip(states, actions):
    state_traj = np.array(state_traj)
    action_traj = np.array(action_traj)
    
    if state_traj.shape[0] != action_traj.shape[0] :
        # Append action 0 to the end of action_traj
        action_traj = np.append(action_traj, 0)

    assert state_traj.shape[0] == action_traj.shape[0]
    
    # link_ids = state_traj[:, :-1].astype(int) 
    link_ids =[[int(link)] for link in state_traj[:, 0]] # link_id, speed, roadtype, time_range 
    speeds = state_traj[:, 1].reshape(-1, 1)/46
    roadtypes = state_traj[:, 2].reshape(-1, 1)
    if np.isnan(roadtypes).any():
        continue  # Skip the entire trajectory if there's a NaN value in roadtypes
    one_hot_roadtypes = one_hot_encode_road_types(roadtypes)
    time_ranges = state_traj[:, 3].reshape(-1, 1)
     # Get the link ID of the last step in the trajectory directly
    last_step_link_id = state_traj[-1, 0].astype(int)

    one_hot_encoded = one_hot_encode_link_ids(link_ids, num_dimensions)
    
    if None in link_ids:
        print("Warning: Some (u, v) pairs were not found in link_mapping.")
    
    # Use link_ids instead of node pairs for further computations
    state_traj = one_hot_encoded

   # Create a column vector filled with this link ID, with the same number of rows as `state_traj`
    last_step_feature = np.full((len(state_traj), 1), last_step_link_id/ 427.0)

    # Append the new feature column to state_traj
    state_traj = np.hstack((one_hot_encoded, last_step_feature))
 
    # Reshape action_traj to make it a column vector
    action_traj = action_traj.reshape(-1, 1)

    # Concatenate state_traj and action_traj along the second axis
    concated_array = np.hstack((state_traj, speeds,one_hot_roadtypes,time_ranges, action_traj))

    demon_list.append(concated_array)

    print('demon_list',demon_list[0])
    print('demon_list',len(demon_list[0][0]))


with open("data/demon.pkl", "wb") as f:
    pickle.dump(demon_list, f)

valid_trajectories = [traj for traj in demon_list if all(np.isin(traj[:, -1], [0, 1, 2, 3]))]

train_list, test_list = train_test_split(valid_trajectories, test_size=0.2, random_state=42)

def reverse_one_hot(one_hot_encoded_array):
    # Assuming the first 427 columns are the one-hot encoded link IDs
    return np.argmax(one_hot_encoded_array[:, :427], axis=1)

# Extract only the link ID and action from train_list
train_trajectories = [np.column_stack((reverse_one_hot(traj), traj[:, -1])) for traj in train_list]

# Extract only the link ID and action from test_list
test_trajectories = [np.column_stack((reverse_one_hot(traj), traj[:, -1])) for traj in test_list]

# Save the extracted trajectories
with open("data/link#/demon_train.pkl", "wb") as f:
    pickle.dump(train_trajectories, f)

with open("data/link#/demon_test.pkl", "wb") as f:
    pickle.dump(test_trajectories, f)

# Output to verify sizes
print('Training trajectories length:', len(train_trajectories))
print('Testing trajectories length:', len(test_trajectories))

# print('train_list',train_list[0])
print('train_list', len(train_list))
print('test_list', len(test_list))

with open("data/demon_train.pkl", "wb") as f:
    pickle.dump(train_list, f)

# Convert train_list to a DataFrame
train_df = pd.DataFrame(np.vstack(train_list))

# Define column names for the DataFrame
column_names = [f"feature_{i}" for i in range(train_df.shape[1] - 1)] + ["action"]

# Assign column names to the DataFrame
train_df.columns = column_names

# Save train_df as a CSV file
train_df.to_csv('data/train_data.csv', index=False)

with open("data/demon_test.pkl", "wb") as f:
    pickle.dump(test_list, f)


# Convert train_list to a DataFrame
test_df = pd.DataFrame(np.vstack(test_list))

# Define column names for the DataFrame
column_names = [f"feature_{i}" for i in range(test_df.shape[1] - 1)] + ["action"]

# Assign column names to the DataFrame
test_df.columns = column_names

# Save train_df as a CSV file
test_df.to_csv('data/test_data.csv', index=False)
