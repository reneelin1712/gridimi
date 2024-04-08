import pickle
import pandas as pd 

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