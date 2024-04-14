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

import wandb

with open('data/link#/demon_train.pkl', "rb") as f:
    demon_train = pickle.load(f)

with open('data/link#/demon_test.pkl', "rb") as f:
    demon_test = pickle.load(f)

venv = DummyVecEnv([lambda: NeumaEnv()] * 4)
rollouts = neuma_rollout()


sweep_config = {
  "name" : "my-sweep",
  "method" : "random",
  "parameters" : {
    "total_steps": {"values":[500_000]},
    "seed":{
      "values" : [1]
    },
    "n_epochs_policy" : {
      "values" : [15],  #[20,50,60]
    },
     "n_epochs_disc" : {
      "values" : [15],  #[20,50,60]
    },
    "learning_rate" :{
      # "distribution":"uniform",
      # "min": 0.000055,
      # # "max": 0.000065
      "values": [ 0.00009] # 0.0003,0.0005,0.0009,0.0001,0.009,0.005,0.003,0.001
    },
    "batch_size":{
      "values":[32]#[32,64,128,256]
    },
    "n_steps" :{
      "values":[2048] #[512,1024,2048,4096]
    },
    "clip_range":{
      "values":[0.2]#[0.1,0.2,0.3]
    },
    "gae_lambda":{
      "values" :[0.99] #[0.95,0.96,0.97,0.98,0.99]
    },
    "vf_coef" :{
      "values":[0.5] #[0.5,1.0]
    },
    "ent_coef":{
      "values":[0.2] #[0.5,1.0]
      # "distribution":"uniform",
      # "min": 0.0,
      # "max": 0.01
    },
    "target_kl":{
      # "distribution":"uniform",
      # "min": 0.003,
      # "max": 0.03
      "values":[0.3]
    }

  },
  "metric": {
      "name":"rmse_space",
      "goal":"rmse_space"
  }

}

sweep_id = wandb.sweep(sweep_config, project="o_airl")


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


def train():
    with wandb.init() as run:
        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            # policy = 'MlpLstmPolicy',
            seed = wandb.config.seed,
            learning_rate= wandb.config.learning_rate,
            batch_size = wandb.config.batch_size,
            n_steps= wandb.config.n_steps,
            n_epochs = wandb.config.n_epochs_policy,
            clip_range = wandb.config.clip_range,
            gae_lambda = wandb.config.gae_lambda,
            vf_coef = wandb.config.vf_coef,
            ent_coef = wandb.config.ent_coef,
            target_kl = wandb.config.target_kl,
            verbose=1,
            tensorboard_log=log_dir_gen,
        )

        reward_net = BasicRewardNet(
            venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
        )

        airl_trainer = AIRL(
            demonstrations=rollouts,
            demo_batch_size=wandb.config.n_steps,
            gen_replay_buffer_capacity=wandb.config.n_steps,
            n_disc_updates_per_round=wandb.config.n_epochs_disc,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
            log_dir=log_dir,
            init_tensorboard=True,
        )

        TIMESTEPS = wandb.config.total_steps
        airl_trainer.train(total_timesteps=TIMESTEPS)
        # for i in range(save_points):  
        torch.save(airl_trainer.reward_test, f"{model_dir}/reward")
        learner.save(f"{model_dir}/")

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

        metrics = {'mean_edit_distance_train':mean_edit_distance_train, 'average_bleu_train': mean_bleu_train, "avg_length_ratio_train":avg_length_ratio_train,"jsd_train":jsd_train, "ended_with_zero_train":ended_with_zero_train,\
                  #  'mean_edit_distance':mean_edit_distance, 'average_bleu': mean_bleu, "avg_length_ratio":avg_length_ratio,"jsd":jsd, "ended_with_zero":ended_with_zero
                  }
        
        wandb.log(metrics)

wandb.agent(sweep_id, function=train, count=1)
