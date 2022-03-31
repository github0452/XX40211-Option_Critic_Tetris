import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris import TetrisEnv
from option_critic import OptionCritic


checkpoint="logs/tetris_fixed/PPO/_model_50000_steps.zip"

# loading model
model = PPO.load(checkpoint)

env = TetrisEnv(board_size=(10,10), grouped_actions=True, only_squares=True, no_rotations=True, max_steps=500)

game_over = False
obs = env.reset()
# env.measure_step_time(verbose=True)
env.render(wait_sec=1, mode='image', verbose=True)
while not game_over:
    action, _states = model.predict(obs, deterministic=False)
    print("Action", action)
    obs, reward, game_over, info = env.step(action)
    env.render(wait_sec=1, mode='image', verbose=True)
    print("Reward", reward)
