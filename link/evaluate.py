import editdistance
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.meteor_score import meteor_score
from nltk.translate.meteor_score import single_meteor_score

import numpy as np
import pickle
import torch
from env import NeumaEnv
from env_test import NeumaEnvTest
from stable_baselines3 import PPO
import pandas as pd


# with open('data/link#/demon_train.pkl', "rb") as f:
#     demon_train = pickle.load(f)

# # Load the CSV data into a DataFrame
# action_qty_df = pd.read_csv('/Users/potato/Documents/IL/linkpred/data/prep/link_actions.csv')
# action_qty_dict = pd.Series(action_qty_df.action_qty.values, index=action_qty_df.link_id).to_dict()


def calculate_edit_distance(generated_seq, reference_seqs):
    edit_distances = [editdistance.eval(generated_seq, ref) for ref in reference_seqs]
    return min(edit_distances) if edit_distances else float('inf')


def create_distribution(lengths):
    max_length = max(lengths)
    distribution = [0] * (max_length + 1)
    for length in lengths:
        distribution[length] += 1

    return np.array(distribution) / len(lengths)

def calculate_hypothetical_rewards(obs, action, model_reward, env, link_id):
    num_actions = action_qty_dict.get(link_id, 1) + 1
    hypothetical_rewards = {}
    for hypothetical_action in range(num_actions):
        obs_next_, _, done, _, info = env.step(hypothetical_action)
        if isinstance(obs, torch.Tensor):
            obs_next_ = obs_next_.numpy()
        if hypothetical_action != action:  # Skip the actual action
            one_hot_hypothetical_action = torch.zeros(4, dtype=torch.float32)
            one_hot_hypothetical_action[hypothetical_action] = 1.0
            hypothetical_reward = model_reward(
                torch.from_numpy(obs).unsqueeze(dim=0).float(),
                one_hot_hypothetical_action.unsqueeze(dim=0).float(),
                torch.from_numpy(obs_next_).unsqueeze(dim=0).float(),  # Placeholder for next_obs
                torch.FloatTensor([False]).unsqueeze(dim=0).float()  # Placeholder for done
            ).item()
            
            # Simulate getting the next link ID for the hypothetical action
            # This part needs actual implementation based on how you can obtain the next link ID
            # For the purpose of this example, it's set to None
            hypothetical_link_id_next = decode_one_hot_to_link_id(obs_next_[:427])
            
            
            hypothetical_rewards[hypothetical_action] = {
                'reward': hypothetical_reward,
                'link_id_next': hypothetical_link_id_next
            }
    return hypothetical_rewards


def group_demonstrations_by_first_link(demon_list):
    grouped_demonstrations = {}
    for traj in demon_list:
        first_link = traj[0, 0]  # Assuming the first column is the link
        if first_link not in grouped_demonstrations:
            grouped_demonstrations[first_link] = []
        grouped_demonstrations[first_link].append(traj)
    return grouped_demonstrations

def decode_one_hot_to_link_id(one_hot_encoded):
    link_id = np.where(one_hot_encoded == 1.0)
    return link_id[0]


def evaluate(model, env, demon_list):
    demonstrations_grouped = group_demonstrations_by_first_link(demon_list)

    all_bleu_scores = []

    length_ratios = []
    demo_lengths = []
    generated_lengths = []

    all_edit_distances = []
    meteor_scores = []

    count_ended_with_zero = 0

    data = []
    
    for idx, traj in enumerate(demon_list):
        obs, _ = env.reset(traj_idx=idx)
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()
        link_id_encoded = obs[0:427]
        link_id = decode_one_hot_to_link_id(link_id_encoded)
        generated_actions = []
        generated_links = [int(link_id)]

        # des=round(obs[427]*427)
        # ori = link_id
        
        while True:
            action, _ = model.predict(obs, deterministic=False)
            # print('action evaluate',action)
            action = int(action)
            if action !=0:
                generated_actions.append(action)
                obs_next, _, done, _, info = env.step(action)

                if isinstance(obs_next, torch.Tensor):
                    obs_next = obs_next.numpy()
                link_id_encoded = obs_next[0:427]
                link_id = decode_one_hot_to_link_id(link_id_encoded)
                if link_id and not done:
                    generated_links.append(int(link_id))

                one_hot_actual_action = torch.zeros(4, dtype=torch.float32)
                one_hot_actual_action[action] = 1.0

                obs = obs_next

            if action ==0 :
                generated_actions.append(action)
                count_ended_with_zero += 1
                done = True
            if done:
                break    
        
        # Find matching demonstrations
        print(f'{traj}, generated_actions', generated_actions)
        print('generated_links',generated_links)
        first_link = generated_links[0] 
        matching_demonstrations = demonstrations_grouped.get(first_link, [])

        # Calculate BLEU score with all matching demonstrations
        references = [list(match_traj[:, 0]) for match_traj in matching_demonstrations]
        # print('init references', references)
      

        if references:
            bleu_score = sentence_bleu(references, generated_links)
            all_bleu_scores.append(bleu_score)

            edit_distance = calculate_edit_distance(generated_links, references)
            all_edit_distances.append(edit_distance / len(traj)) # Normalize by the length of the trajectory

        else:
            # Handle case where there are no matching references
            print(f"No matching demonstrations found for first link {first_link}")
            # all_edit_distances.append(float('inf')) 
            all_edit_distances.append(0) 
            all_bleu_scores.append(0)  # No BLEU score can be calculated

        demon_links = list(traj[:,0].flatten())
        demo_length = len(demon_links)
        generated_length = len(generated_links)
        
        if demo_length != 0:
            length_ratio = generated_length / demo_length
            length_ratios.append(length_ratio)

        demo_lengths.append(demo_length)
        generated_lengths.append(generated_length)

    mean_edit_distance = np.mean(all_edit_distances) if all_edit_distances else float('inf')
    mean_bleu = np.mean(all_bleu_scores)
    avg_length_ratio = np.mean(length_ratios)

    demo_distribution = create_distribution(demo_lengths)
    gen_distribution = create_distribution(generated_lengths)

    max_length = max(len(demo_distribution), len(gen_distribution))
    demo_distribution = np.pad(demo_distribution, (0, max_length - len(demo_distribution)), 'constant')
    gen_distribution = np.pad(gen_distribution, (0, max_length - len(gen_distribution)), 'constant')

    jsd = distance.jensenshannon(demo_distribution, gen_distribution)

    # mean_meteor = np.mean(meteor_scores) if meteor_scores else 0
    mean_meteor = None

    ended_with_zero = (count_ended_with_zero / len(demon_list))

    return mean_edit_distance, mean_bleu,avg_length_ratio, jsd, ended_with_zero, mean_meteor


# # load model
# AIRL
# model_path =  "models/airl/AIRL-2024-02-25-19-34-21-/policy" #AIRL-2024-02-24-20-23-21
# model = PPO.load(model_path)
# model_path_reward = "models/airl/AIRL-2024-02-25-19-34-21-/reward" 
# model_reward = torch.load(model_path_reward)
# model_reward.eval()

# GAIL 
# model_path = "models/gail/GAILL-2024-01-04-23-22-28"
# model = PPO.load(model_path)

# policy = model.policy
# print(policy)


# BC
# model_path = "models/BC-2024-02-24-19-19-43/policy_model.pt"
# model = torch.load(model_path)
# model.eval()


# # Load the LSTM model
# model_path = "models/LSTM-2024-01-04-23-52-21/lstm_model.pt"  # Update this path

# # Model parameters
# input_size = 1  # Depends on your state size
# hidden_layer_size = 100  # Example size
# # output_size = actions.shape[1]  # Depends on your action size
# output_size = 1  # Set to 1 as each action is a single value
# # Initialize the model with the same parameters
# model = LSTMModel(input_size, hidden_layer_size, output_size)
# model.load_state_dict(torch.load(model_path))
# model.eval()  # Set the model to evaluation mode

# mean_edit_distance_train, mean_bleu_train,avg_length_ratio_train, jsd_train, ended_with_zero_train, mean_meteor_train= evaluate(model,NeumaEnv(),demon_train)
# print(f'mean_edit_distance_train:{mean_edit_distance_train}')
# print(f'mean_bleu_train:{mean_bleu_train}')
# print(f'avg_length_ratio_train:{avg_length_ratio_train}')
# print(f'jsd_train:{jsd_train}')
# print(f'ended_with_zero_train:{ended_with_zero_train}')
# print(f'meteor_train:{mean_meteor_train}')

# mean_edit_distance, mean_bleu,avg_length_ratio, jsd, ended_with_zero, mean_meteor= evaluate(model,NeumaEnvTest(),demon_test)
# print(f'mean_edit_distance:{mean_edit_distance}')
# print(f'mean_bleu:{mean_bleu}')
# print(f'avg_length_ratio:{avg_length_ratio}')
# print(f'jsd:{jsd}')
# print(f'ended_with_zero:{ended_with_zero}')
# print(f'meteor:{mean_meteor}')


