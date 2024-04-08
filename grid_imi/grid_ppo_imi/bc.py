import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

from stable_baselines3 import PPO

rng = np.random.default_rng(0)
# env = gym.make('GridWorld-v0',grid_size=7)

# def make_grid_env():
#     return gym.make('GridWorld-v0', grid_size=7)
# # Create a vectorized environment
# vec_env = make_vec_env(make_grid_env, n_envs=4, rng=rng)

vec_env = make_vec_env(
    env_name="GridWorld-v0",  # Environment ID as a string
    n_envs=4,               # Number of parallel environments
    rng=rng,                # Random number generator
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]  # for computing rollouts
)

# Load the trained model
model = PPO.load("models/ppo_grid")

rollouts = rollout.rollout(
    model,
    vec_env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    demonstrations=transitions,
    rng=rng,

)
bc_trainer.train(n_epochs=100)
bc_trainer.policy.save("models/bc_policy")
reward, _ = evaluate_policy(bc_trainer.policy, vec_env, 10)
print("Reward:", reward)

