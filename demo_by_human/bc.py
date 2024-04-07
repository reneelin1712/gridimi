import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

from stable_baselines3 import PPO
from rollout_expert import human_rollout

rng = np.random.default_rng(0)

vec_env = make_vec_env(
    env_name="GridWorld-v0",  # Environment ID as a string
    n_envs=4,               # Number of parallel environments
    rng=rng,                # Random number generator
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]  # for computing rollouts
)

rollouts = human_rollout()
# transitions = rollout.flatten_trajectories(rollouts) #TODO

bc_trainer = bc.BC(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    demonstrations=rollouts,
    rng=rng,
    # optimizer_kwargs={
    #     'lr': 0.0003,
    # },
    # l2_weight=1e-5, #TODO
    # batch_size = 64

)
bc_trainer.train(n_epochs=100)
bc_trainer.policy.save("models/bc_policy")
reward, _ = evaluate_policy(bc_trainer.policy, vec_env, 10)
print("Reward:", reward)

