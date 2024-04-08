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
import ray
from ray import tune
from ray.air.integrations.wandb import setup_wandb


with open('data/link#/demon_train.pkl', "rb") as f:
    demon_train = pickle.load(f)

with open('data/link#/demon_test.pkl', "rb") as f:
    demon_test = pickle.load(f)

def train_function_wandb(config):
    with wandb.init(config=config, project="link_ray") as run:

      venv = DummyVecEnv([lambda: NeumaEnv()] * 4)
      rollouts = neuma_rollout()

      learner = PPO(
          env=venv,
          policy=MlpPolicy,
          seed=config["seed"],
          learning_rate=config["learning_rate"],
          batch_size=config["batch_size"],
          n_steps=config["n_steps"],
          n_epochs=config["n_epochs_policy"],
          clip_range=config["clip_range"],
          gae_lambda=config["gae_lambda"],
          vf_coef=config["vf_coef"],
          ent_coef=config["ent_coef"],
          target_kl=config["target_kl"],
          verbose=1,
      )

      reward_net = BasicRewardNet(
          venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
      )

      airl_trainer = AIRL(
          demonstrations=rollouts,
          demo_batch_size=config["n_steps"],
          gen_replay_buffer_capacity=config["n_steps"],
          n_disc_updates_per_round=config["n_epochs_disc"],
          venv=venv,
          gen_algo=learner,
          reward_net=reward_net,
          allow_variable_horizon=True,
      )

      TIMESTEPS = config["total_steps"]
      airl_trainer.train(total_timesteps=TIMESTEPS)

      today = date.today()
      t = time.localtime()
      current_time = time.strftime("%H-%M-%S", t)
      model_dir = f"models/airl/AIRL-{today}-{current_time}"
      if not os.path.exists(model_dir):
          os.makedirs(model_dir)

      torch.save(airl_trainer.reward_test, f"{model_dir}/reward")
      learner.save(f"{model_dir}/")

      mean_edit_distance_train, mean_bleu_train, avg_length_ratio_train, jsd_train, ended_with_zero_train, mean_meteor_train = evaluate(learner.policy, NeumaEnv(), demon_train)
      mean_edit_distance, mean_bleu, avg_length_ratio, jsd, ended_with_zero, mean_meteor = evaluate(learner.policy, NeumaEnvTest(), demon_test)

      metrics = {
          'mean_edit_distance_train': mean_edit_distance_train,
          'average_bleu_train': mean_bleu_train,
          "avg_length_ratio_train": avg_length_ratio_train,
          "jsd_train": jsd_train,
          "ended_with_zero_train": ended_with_zero_train,
          "meteor_train": mean_meteor_train,
          'mean_edit_distance': mean_edit_distance,
          'average_bleu': mean_bleu,
          "avg_length_ratio": avg_length_ratio,
          "jsd": jsd,
          "ended_with_zero": ended_with_zero,
          "meteor": mean_meteor,
      }

      wandb.log(metrics)
      return {"average_bleu_train":  mean_bleu_train}

def tune_with_setup():
    tuner = tune.Tuner(
        train_function_wandb,
        tune_config=tune.TuneConfig(
            metric="average_bleu_train",
            mode="max",
            num_samples=100,
        ),
        param_space={
            "total_steps": 2_000_000,
            "seed": 1,
            "n_epochs_policy": tune.randint([10, 20]),
            "n_epochs_disc": tune.randint([10,20]),
            "learning_rate": tune.uniform(0.000001, 0.003),
            "batch_size": tune.choice([16,32,64,128,512]),
            "n_steps": tune.choice([1024, 2048]),
            "clip_range": tune.choice([0.1,0.2,0.3]),
            "gae_lambda": tune.choice([0.95,0.96,0.97,0.98,0.99]),
            "vf_coef": tune.choice([0.5,1.0]),
            "ent_coef": 0.2,
            "target_kl": 0.3,
        },
    )
    tuner.fit()

if __name__ == "__main__":
    ray.init()
    tune_with_setup()