import pickle
import pandas as pd 
import numpy as np

# states = np.load('data/traj_20181029_dX_0830_0900_states_sameOD.npy', allow_pickle=True)

# for state_traj in states[900:903]:
#     print('state_traj',state_traj)
#     state_traj = np.array(state_traj)


with open('data/link#/demon_train.pkl', "rb") as f:
    demon_test = pickle.load(f)

# Prepare data for DataFrame
data = []
for episode_idx, episode in enumerate(demon_test):
    for obs, action in episode:
        data.append([episode_idx, obs, action])

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=["Episode", "Observation", "Action"])

# Save the DataFrame to an Excel file
excel_filename = 'data/demonstration_data.xlsx'
df.to_excel(excel_filename, index=False)