from stable_baselines3.ppo import MlpPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle

with open('data/demon_train.pkl', "rb") as f:
    demon_list = pickle.load(f)

class ExpertEnv(gym.Env):
    
    def __init__(self):
        super(ExpertEnv, self).__init__()
        
        # Define action space. Neighbors 1,2,3; 0 is stop
        self.action_space = spaces.Discrete(4) 

        # Define observation space
        self.num_features = 433
        low = np.full(self.num_features, 0.0)
        high = np.full(self.num_features, 1.0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = spaces.Discrete(427) # TODO: this will lead to shape error?
      
        # Initialize other variables
        self.traj_idx =0
        # self.reset()  # the init will call reset by default
        self.max_steps = 16
        self.done = False
        self.reward = 0

    def reset(self,seed=None,options=None):
        self.done = False
        self.reward = 0

        if self.traj_idx == len(demon_list)-1:
            self.traj_idx =0

        self.episode_len = len(demon_list[self.traj_idx])
        self.counter = 0
        self.state = demon_list[self.traj_idx][0][0:self.num_features]

        # print('init state',self.state)

        return np.array(self.state).astype(np.float32), {}


    def step(self, action):
        # expert_action = demon_list[self.traj_idx][self.counter][-1]

        if self.counter == self.episode_len-1:
            pass
        else:
            next_state = demon_list[self.traj_idx][self.counter+1][0:self.num_features]
            next_state = np.array(next_state, dtype=np.float32) 
            self.state = next_state
        
        if self.counter == self.episode_len-1:
            self.traj_idx +=1
            self.reset()
            
            done = True
        else:
            done = False

        reward = 1.0

        info = {}

        
        self.counter +=1
        truncated = None
        
        # print('self.state',self.state)
        return np.array(self.state).astype(np.float32), reward, done,truncated, info
    
    def render(self, mode='console'):
      pass

    def close(self):
      temp = demon_list[self.traj_idx][self.counter][-1]
      
      return np.array(temp).astype(np.float32)
    
