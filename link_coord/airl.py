from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from env import NeumaEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from rollout_expert import neuma_rollout
import torch
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import os
from datetime import date
import time
import pickle
import numpy as np

from evaluate import evaluate

from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout

from env_test import NeumaEnvTest

from stable_baselines3.common.buffers import RolloutBuffer
from imitation.algorithms import bc

with open('./data/link#/demon_train.pkl', "rb") as f:
    demon_train = pickle.load(f)

with open('./data/link#/demon_test.pkl', "rb") as f:
    demon_test = pickle.load(f)

rng = np.random.default_rng(0)

venv = DummyVecEnv([lambda: NeumaEnv()] * 4)
rollouts = neuma_rollout()

today = date.today()
t = time.localtime()
current_time = time.strftime("%H-%M-%S", t)
model_dir = f"models/airl/AIRL-{today}-{current_time}"
log_dir = f"logs/AIRL-{current_time}"
log_dir_gen = f"logs/PPO"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(log_dir_gen):
    os.makedirs(log_dir_gen)


# BC
# bc_policy_path = "models/BC-2024-04-01-22-52-12/policy_model.pt"
# bc_policy = torch.load(bc_policy_path)
# torch.save(bc_trainer.policy.state_dict(), bc_policy_path)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    # policy = 'MlpLstmPolicy',
    seed=1,
    n_epochs=15,
    learning_rate=0.00003,
    batch_size=16,
    n_steps=2048,
    clip_range=0.2,
    gae_lambda=0.99,
    vf_coef=0.5,
    ent_coef=0.2,
    target_kl=0.02,
    verbose=1,
    tensorboard_log=log_dir_gen,
    # _init_setup_model=False,
)

# learner.policy = bc_policy
# learner.policy.load_state_dict(torch.load(bc_policy_path))

# learner.rollout_buffer = RolloutBuffer(
#     buffer_size=learner.n_steps,
#     observation_space=learner.observation_space,
#     action_space=learner.action_space,
#     device=learner.device,
#     gamma=learner.gamma,
#     gae_lambda=learner.gae_lambda,
#     n_envs=learner.n_envs,
# )

reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)


airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=15,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
    log_dir=log_dir,
    init_tensorboard=True,
)

TIMESTEPS = 500_000 #1_000_000
# save_points = 5 # TODO: how to save several intermedian models


airl_trainer.train(total_timesteps=TIMESTEPS)
# for i in range(save_points):  
torch.save(airl_trainer.reward_test, f"{model_dir}/reward")
learner.save(f"{model_dir}/")

# mean_edit_distance, mean_bleu,avg_length_ratio, jsd, ended_with_zero, mean_meteor= evaluate(learner.policy,NeumaEnv(),demon_test)
# print(f'mean_edit_distance:{mean_edit_distance}')
# print(f'mean_bleu:{mean_bleu}')
# print(f'avg_length_ratio:{avg_length_ratio}')
# print(f'jsd:{jsd}')
# print(f'ended_with_zero:{ended_with_zero}')
# print(f'meteor:{mean_meteor}')

mean_edit_distance_train, mean_bleu_train,avg_length_ratio_train, jsd_train, ended_with_zero_train, mean_meteor_train= evaluate(learner.policy,NeumaEnv(),demon_train)
print(f'mean_edit_distance_train:{mean_edit_distance_train}')
print(f'mean_bleu_train:{mean_bleu_train}')
print(f'avg_length_ratio_train:{avg_length_ratio_train}')
print(f'jsd_train:{jsd_train}')
print(f'ended_with_zero_train:{ended_with_zero_train}')
print(f'meteor_train:{mean_meteor_train}')

# mean_edit_distance, mean_bleu,avg_length_ratio, jsd, ended_with_zero, mean_meteor= evaluate(learner.policy,NeumaEnvTest(),demon_test)
# print(f'mean_edit_distance:{mean_edit_distance}')
# print(f'mean_bleu:{mean_bleu}')
# print(f'avg_length_ratio:{avg_length_ratio}')
# print(f'jsd:{jsd}')
# print(f'ended_with_zero:{ended_with_zero}')
# print(f'meteor:{mean_meteor}')

