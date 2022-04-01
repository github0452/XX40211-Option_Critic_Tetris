import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris import TetrisEnv
from option_critic import OptionCritic


# checkpoint="logs/tetris_fixed/PPO/_model_50000_steps.zip"
folder = "logs/tetris_fixed/OptionCritic/"
checkpoint="logs/tetris_fixed/OptionCritic/best_model.zip"
env = TetrisEnv(board_size=(6,6), grouped_actions=True, only_squares=True, no_rotations=True)

# loading model
# model = PPO.load(checkpoint)
model = OptionCritic(env, update_frequency=1, logdir=folder)
model.load(checkpoint)

game_over      = False
obs            = env.reset()
current_option = None
# env.measure_step_time(verbose=True)
env.render(wait_sec=1, mode='image', verbose=True)
while not game_over:
    # action, _ = model.predict(obs, deterministic=False)
    # print("Action", action)
    action, current_option = model.predict(obs, deterministic=False, current_option=current_option) # option critic version
    print(f"Action: {action}| option: {current_option}")
    obs, reward, game_over, info = env.step(action)
    env.render(wait_sec=1, mode='image', verbose=True)
    print("Reward", reward)
