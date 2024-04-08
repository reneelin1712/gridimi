import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

TOTAL_STEPS = 300000
# Parallel environments
# vec_env = make_vec_env("GridWorld-v0", n_envs=4)
env = gym.make('GridWorld-v0',grid_size=7)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TOTAL_STEPS)
model.save(f"models/ppo_grid_{TOTAL_STEPS}")
