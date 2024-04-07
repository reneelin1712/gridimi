import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

from rollout_expert import human_rollout

SEED = 42

env = make_vec_env(
    env_name="GridWorld-v0",  # Environment ID as a string
    n_envs=4,               # Number of parallel environments
    rng=np.random.default_rng(SEED),                # Random number generator
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]  # for computing rollouts
)

rollouts = human_rollout()


learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,          # Consider reducing for more frequent updates
    ent_coef=0.2,          # Increase entropy coefficient
    learning_rate=0.0005,  
    gamma=0.95,             
    clip_range=0.2,         # Increasing clip range
    vf_coef=0.1,
    n_epochs=16,            # Increase number of epochs
    seed=SEED,
)

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048, 
    gen_replay_buffer_capacity=1024, #1024 is better than 512
    n_disc_updates_per_round=16, #when policy is 10, this is 5 is bad; tyring both on 20
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True #TODO
)


# evaluate the learner before training
env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# train the learner and evaluate again
airl_trainer.train(800_000)  # Train for 800_000 steps to match expert.
# Save the trained policy
learner.save("models/airl_policy")

env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))