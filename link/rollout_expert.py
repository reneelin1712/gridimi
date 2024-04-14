from env_expert import ExpertEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import pickle

with open('data/demon_train.pkl', "rb") as f:
    demon_list = pickle.load(f)

rng = np.random.default_rng(0)

def expert_demo_generator(demon_list):
    while True:  # Loop to restart from the beginning when all demos are exhausted
        for episode in demon_list:
            for step_data in episode:
                obs, action = step_data[0:-1], step_data[-1]
                yield obs, action

# Initialize the expert demonstration generator
expert_demos = expert_demo_generator(demon_list)

def expert_policy_callable(obs, states=None, episode_starts=None):
    # Return the next action for each observation in obs
    actions = [next(expert_demos)[1] for _ in obs]
    return np.array(actions), None

def neuma_rollout():
    expeEnv = ExpertEnv()
    rollouts = rollout.rollout(
        expert_policy_callable,
        DummyVecEnv([lambda: RolloutInfoWrapper(expeEnv)] * 1),
        rollout.make_sample_until(min_timesteps=None, min_episodes=6492),
        rng=rng
    )

    # # Prepare data for DataFrame
    # data = []
    # for episode_idx, episode in enumerate(rollouts):
    #     for obs, act in zip(episode.obs, episode.acts):
    #         data.append([episode_idx, obs[0], act])
    # return rollouts, data

    return rollouts


# _, data = neuma_rollout()

# df = pd.DataFrame(data, columns=["Episode", "Observation", "Action"])

# excel_filename = 'data/rollouts_data.xlsx'
# df.to_excel(excel_filename, index=False)



