from env import NeumaEnv
from env_test import NeumaEnvTest
from stable_baselines3.common.vec_env import DummyVecEnv
from rollout_expert import neuma_rollout
import torch

import os
from datetime import date
import time
import pickle
from imitation.algorithms import bc
import numpy as np

import torch.optim as optim

from evaluate import evaluate
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from typing import Callable

with open('data/link#/demon_train.pkl', "rb") as f:
    demon_train = pickle.load(f)

with open('data/link#/demon_test.pkl', "rb") as f:
    demon_test = pickle.load(f)

venv = DummyVecEnv([lambda: NeumaEnv()] * 4)
rollouts = neuma_rollout()
rng = np.random.default_rng(0)

today = date.today()
t = time.localtime()
current_time = time.strftime("%H-%M-%S", t)


def train():
    model_dir = f"models/BC-{today}-{current_time}"  
    log_dir = f"logs/BC-{current_time}" 

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    lr_schedule = linear_schedule(initial_value=0.001)

    policy = ActorCriticPolicy(
        lr_schedule=lr_schedule,
        observation_space=venv.observation_space, 
        action_space=venv.action_space)

    bc_trainer = bc.BC(
      policy=policy,
      observation_space=venv.observation_space,
      action_space=venv.action_space,
      demonstrations=rollouts,
      rng=rng,
      optimizer_kwargs={
        'lr': 0.0003,
    },
    l2_weight=1e-5,
    batch_size = 32
    )
    
    bc_trainer.train(n_epochs=30)
    torch.save(bc_trainer.policy, f"{model_dir}/policy_model.pt")

    mean_edit_distance_train, mean_bleu_train,avg_length_ratio_train, jsd_train, ended_with_zero_train, mean_meteor_train= evaluate(bc_trainer.policy,NeumaEnv(),demon_train)
    print(f'mean_edit_distance_train:{mean_edit_distance_train}')
    print(f'mean_bleu_train:{mean_bleu_train}')
    print(f'avg_length_ratio_train:{avg_length_ratio_train}')
    print(f'jsd_train:{jsd_train}')
    print(f'ended_with_zero_train:{ended_with_zero_train}')
    print(f'meteor_train:{mean_meteor_train}')

    # mean_edit_distance, mean_bleu,avg_length_ratio, jsd, ended_with_zero, mean_meteor= evaluate(bc_trainer.policy,NeumaEnvTest(),demon_test)
    # print(f'mean_edit_distance:{mean_edit_distance}')
    # print(f'mean_bleu:{mean_bleu}')
    # print(f'avg_length_ratio:{avg_length_ratio}')
    # print(f'jsd:{jsd}')
    # print(f'ended_with_zero:{ended_with_zero}')
    # print(f'meteor:{mean_meteor}')

train()
