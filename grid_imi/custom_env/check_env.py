from stable_baselines3.common.env_checker import check_env
from gridEnv import GridEnv

env = GridEnv(grid_size=6)
# It will check your custom environment and output additional warnings if needed
check_env(env)