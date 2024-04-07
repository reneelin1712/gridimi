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

import wandb

import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

def train_function_wandb(config):
    with wandb.init(config=config, project="grid_ray") as run:

    # for i in range(3):
      env = make_vec_env(
          env_name="GridWorld-v0",
          n_envs=4,
          rng=np.random.default_rng(config["seed"]),
          post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]
      )

      rollouts = human_rollout()

      learner = PPO(
          env=env,
          policy=MlpPolicy,
          batch_size=config["batch_size"],
          ent_coef=0.2,
          learning_rate=config["learning_rate"],
          gamma=0.95,
          clip_range=0.2,
          vf_coef=0.1,
          n_epochs=config["n_epochs_policy"],
          seed=config["seed"],
      )

      reward_net = BasicRewardNet(
          observation_space=env.observation_space,
          action_space=env.action_space,
          normalize_input_layer=RunningNorm,
      )

      airl_trainer = AIRL(
          demonstrations=rollouts,
          demo_batch_size=2048,
          gen_replay_buffer_capacity=1024,
          n_disc_updates_per_round=config["n_epochs_disc"],
          venv=env,
          gen_algo=learner,
          reward_net=reward_net,
          allow_variable_horizon=True
      )

      env.seed(config["seed"])
      learner_rewards_before_training, _ = evaluate_policy(
          learner, env, 100, return_episode_rewards=True,
      )

      airl_trainer.train(config["total_steps"])
      learner.save("models/airl_policy")

      env.seed(config["seed"])
      learner_rewards_after_training, _ = evaluate_policy(
          learner, env, 100, return_episode_rewards=True,
      )

      metrics = {
    'mean_reward_after_training': np.mean(learner_rewards_after_training),
    'mean_reward_before_training': np.mean(learner_rewards_before_training)
}
      wandb.log(metrics)
      # train.report({'mean_reward_after_training': np.mean(learner_rewards_after_training)})
      return {"mean_reward_after_training":  np.mean(learner_rewards_after_training)}

def tune_with_setup():
    tuner = tune.Tuner(
        train_function_wandb,
        tune_config=tune.TuneConfig(
            metric="mean_reward_after_training",
            mode="max",
            num_samples=3,
        ),
        param_space={
            "total_steps": tune.grid_search([300_000]),
            "seed": tune.grid_search([42]),
            "n_epochs_policy": tune.grid_search([15, 20]),
            "n_epochs_disc": tune.grid_search([15,20]),
            "learning_rate": tune.grid_search([0.0003, 0.0001,0.0005]),
            "batch_size": tune.grid_search([32,64]),
            "n_steps": tune.grid_search([512, 1024, 2048, 4096]),
            "clip_range": tune.grid_search([0.2]),
            "gae_lambda": tune.grid_search([0.99]),
            "vf_coef": tune.grid_search([0.5]),
            "ent_coef": tune.grid_search([0.2]),
            "target_kl": tune.grid_search([0.3])
        },
    )
    tuner.fit()


tune_with_setup()
