import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import pickle
import random

transit = pd.read_csv('./data/prep/transit_20181029_dX_0830_0900.csv')

with open('data/demon_test.pkl', "rb") as f:
    demon_list = pickle.load(f)

link_speeds = pd.read_csv('./data/prep/link_speeds_20181029_dX_0830_0900.csv')
link_speeds_dict = {(row['link_id'], row['100s_interval']): row['link_speed']
                    for index, row in link_speeds.iterrows()}

class NeumaEnvTest(gym.Env):
    
    def __init__(self):
        super(NeumaEnvTest, self).__init__()

        # Define action space. Neighbors 1,2,3; 0 is stop
        self.action_space = spaces.Discrete(4) 

        # Define observation space
        self.num_features = 433
        low = np.full(self.num_features, 0.0)
        high = np.full(self.num_features, 1.0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = spaces.Discrete(427) # TODO: this lead to poor result
      
        # Initialize other variables
        self.traj_idx =0
        self.reset()
        self.max_steps = 16
        self.done = False
        self.reward = 0

    def one_hot_encode_link_id(self,link_id, num_dimensions):
        one_hot_encoded = np.zeros(num_dimensions)
        one_hot_encoded[int(link_id)] = 1.
        return one_hot_encoded
    
    def decode_one_hot_to_link_id(self,one_hot_encoded):
        link_id = np.where(one_hot_encoded == 1.0)
        return int(link_id[0])
    

    def one_hot_encode_road_type(self,road_type):
        one_hot_encoded = np.zeros(4)
        one_hot_encoded[int(road_type)] = 1.
        return one_hot_encoded


    def reset(self,traj_idx=None,seed=None,options=None):
        self.done = False
        self.reward = 0
        self.current_step = 0

        # If a trajectory index is provided, use it, otherwise choose a random one.
        if traj_idx is not None:
            self.traj_idx = traj_idx
        else:
            self.traj_idx = random.randint(0, len(demon_list)-1)

        self.episode_len = len(demon_list[self.traj_idx])
        self.counter = 0
        self.state = demon_list[self.traj_idx][0][0:self.num_features]

        self.timeatfirst = demon_list[self.traj_idx][0][-2]

        return np.array(self.state).astype(np.float32),{}


    def step(self, action):
        self.current_step += 1
        link_id = self.decode_one_hot_to_link_id(self.state[0:427])

        matching_row = transit[(transit['link_id'] == link_id) & (transit['action'] == action)]

        if not matching_row.empty:
            next_link_id = matching_row.iloc[0]['next_link_id']
            next_link_id_encode = self.one_hot_encode_link_id(next_link_id,427)

            feature_428 = self.state[427]
            link_speed = link_speeds_dict.get((next_link_id, self.timeatfirst), 0)/46
            road_type = matching_row.iloc[0]['road_type']
            road_type_encoded = self.one_hot_encode_road_type(road_type)

            self.state = np.hstack((next_link_id_encode, [feature_428], np.array([link_speed]), road_type_encoded))

        else:
            next_link_id = None
            self.done = True

        if self.current_step >= self.max_steps:
            self.done = True

        self.counter += 1
        truncated = None

        return self.state, self.reward, self.done, truncated, {}

