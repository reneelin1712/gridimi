import gymnasium as gym
from stable_baselines3 import PPO

from stable_baselines3.common.policies import ActorCriticPolicy


# Create environment
env = gym.make('GridWorld-v0', grid_size=7)

# Load the trained model
# model = PPO.load("models/ppo_grid")
# model = ActorCriticPolicy.load("models/bc_policy")
# model = PPO.load("models/gail_policy")
model = PPO.load("models/airl_policy")

obs, _ = env.reset()
max_steps = 15

# Run the loop for a limited number of steps
for step in range(max_steps):
    print('obs',obs)
    action, _states = model.predict(obs)
    next_obs, rewards, dones, _, info = env.step(action)
    env.render()

    obs = next_obs
    print('dones',dones)
    if dones:
        print('break')
        break 

env.close()
