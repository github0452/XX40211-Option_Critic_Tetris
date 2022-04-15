import gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
from option_critic.option_critic import OptionCritic


# checkpoint="logs/tetris_fixed/PPO/_model_50000_steps.zip"
# folder = "logs/tetris_fixed/OptionCritic/"
checkpoint="logs_gpu/dqn/PPO/best_model.zip"

action_type = 'grouped'
env_max_step = 18000
reward_type='standard'
reward_scaling=None
board_size = (10,10)
env = TetrisEnv(board_size=board_size, action_type=action_type, reward_type=reward_type, max_steps=env_max_step, reward_scaling=reward_scaling)

# loading model
model = PPO.load(checkpoint,force_reset=False)
# model = OptionCritic(env, update_frequency=1, logdir=folder)
# model.load(checkpoint)

game_over      = False
obs            = env.reset()
current_option = None
# env.measure_step_time(verbose=True)
env.render(wait_sec=1, mode='image', verbose=True)
while not game_over:
    action, _ = model.predict(obs, deterministic=False)
    print("Action", action)
    # action, current_option = model.predict(obs, deterministic=False, current_option=current_option) # option critic version
    # print(f"Action: {action}| option: {current_option}")
    obs, reward, game_over, info = env.step(action)
    env.render(wait_sec=1, mode='image', verbose=True)
    print("Reward", reward)
