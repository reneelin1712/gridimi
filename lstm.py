from stable_baselines3 import PPO
import gymnasium as gym

import torch
import torch.nn as nn

# def generate_data_with_policy(env, model, num_sequences, sequence_length, history_length=3):
#     sequences = []
#     actions = []
#     next_states = []

#     for _ in range(num_sequences):
#         obs, _ = env.reset()
#         history = [obs] * history_length  # Initialize history with the initial observation
#         print('history', history)

#         for _ in range(sequence_length):
#             action, _ = model.predict(obs) #TODO deterministic = True?
#             next_obs, _, _, _, _ = env.step(action)

#             # Update history
#             history.pop(0)
#             history.append(next_obs)

#             # Flatten history to create a state representation
#             flat_history = [item for sublist in history for item in sublist]

#             sequences.append(flat_history)
#             actions.append(action)
#             next_states.append(flat_history[len(obs):])  # Shift to the next state

#             obs = next_obs  # Update the current observation

#     return sequences, actions, next_states

def generate_data_with_policy(env, model, num_sequences, sequence_length, history_length=3):
    sequences = []
    actions = []

    for _ in range(num_sequences):
        obs, _ = env.reset()
        obs = [obs]  # Convert to list
        history = [obs] * history_length  # Initialize history with the initial observation

        for _ in range(sequence_length):
            action, _ = model.predict(obs)  # Assuming obs is in the correct format for predict
            next_obs, _, _, _, _ = env.step(action)
            next_obs = [next_obs]  # Convert to list

            # Update history
            history.pop(0)
            history.append(next_obs)

            # Flatten history to create a state representation
            flat_history = [item for sublist in history for item in sublist]

            sequences.append(flat_history)
            actions.append(action)

            obs = next_obs  # Update the current observation

    return sequences, actions



# Load the trained model
model = PPO.load("models/ppo_grid")

# Generate data
env = gym.make('GridWorld-v0', grid_size=7)
sequences, actions = generate_data_with_policy(env, model, num_sequences=1, sequence_length=12)

# print('seq', sequences)
# print('action',actions)




class GridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GridLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence)
        output = self.linear(lstm_out[-1])  # We only care about the last output
        return output
    

def train_and_save_model(model, sequences, actions, num_epochs=10, model_save_path='lstm_model.pth'):
    loss_function = nn.CrossEntropyLoss()  # Assuming actions are categorical
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0

        for sequence, action in zip(sequences, actions):
            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            
            print('sequence',sequence)
            print('action',action)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).view(-1, 1, len(sequence[0]))
            action_tensor = torch.tensor(action, dtype=torch.long)  # Assuming action is a single integer

            pred_action = model(sequence_tensor)
            loss = loss_function(pred_action, action_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Average Loss: {total_loss / len(sequences)}")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# Example usage
lstm_model = GridLSTM(input_size=1, hidden_size=50, output_size=7)
train_and_save_model(lstm_model, sequences, actions, num_epochs=10, model_save_path='lstm_model.pth')
