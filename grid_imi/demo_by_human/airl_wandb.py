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

sweep_config = {
  "name" : "grid-sweep",
  "method" : "random",
  "parameters" : {
    "total_steps": {"values":[800_000]},
    "seed":{
      "values" : [42]
    },
    "n_epochs_policy" : {
      "values" : [15,20],  #[20,50,60]
    },
     "n_epochs_disc" : {
      "values" : [15],  #[20,50,60]
    },
    "learning_rate" :{
      "values": [ 0.0003] # 0.0003,0.0005,0.0009,0.0001,0.009,0.005,0.003,0.001
    },
    "batch_size":{
      "values":[32]
    },
    "n_steps" :{
      "values":[512,1024,2048,4096] #[512,1024,2048,4096]
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
      "values":[0.2] 
    },
    "target_kl":{
      "values":[0.3]
    }

  },
  "metric": {
      "name":"rmse_space",
      "goal":"rmse_space"
  }

}

sweep_id = wandb.sweep(sweep_config, project="grid_airl")


def train():
    with wandb.init() as run:        

      env = make_vec_env(
          env_name="GridWorld-v0",  # Environment ID as a string
          n_envs=4,               # Number of parallel environments
          rng=np.random.default_rng(wandb.config.seed),                # Random number generator
          post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]  # for computing rollouts
      )

      rollouts = human_rollout()

      learner = PPO(
          env=env,
          policy=MlpPolicy,
          batch_size=wandb.config.batch_size,          # Consider reducing for more frequent updates
          ent_coef=0.2,          # Increase entropy coefficient
          learning_rate=wandb.config.learning_rate,  
          gamma=0.95,             
          clip_range=0.2,         # Increasing clip range
          vf_coef=0.1,
          n_epochs=wandb.config.n_epochs_policy,            # Increase number of epochs
          seed=wandb.config.seed,
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
          n_disc_updates_per_round=wandb.config.n_epochs_disc, #when policy is 10, this is 5 is bad; tyring both on 20
          venv=env,
          gen_algo=learner,
          reward_net=reward_net,
          allow_variable_horizon=True #TODO
      )


      # evaluate the learner before training
      env.seed(wandb.config.seed)
      learner_rewards_before_training, _ = evaluate_policy(
          learner, env, 100, return_episode_rewards=True,
      )

      # train the learner and evaluate again
      airl_trainer.train(wandb.config.total_steps)  # Train for 800_000 steps to match expert.
      # Save the trained policy
      learner.save("models/airl_policy")

      env.seed(wandb.config.seed)
      learner_rewards_after_training, _ = evaluate_policy(
          learner, env, 100, return_episode_rewards=True,
      )

      # print("mean reward after training:", np.mean(learner_rewards_after_training))
      # print("mean reward before training:", np.mean(learner_rewards_before_training))
      metrics = {'mean reward after training':np.mean(learner_rewards_after_training),
                 'mean reward before training':np.mean(learner_rewards_before_training)}
      wandb.log(metrics)


wandb.agent(sweep_id, function=train, count=1)