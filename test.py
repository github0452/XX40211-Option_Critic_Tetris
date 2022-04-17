import gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
from option_critic.option_critic import OptionCritic


# checkpoint="logs/tetris_fixed/PPO/_model_50000_steps.zip"
folder = "logs/tetris_fixed/OptionCritic/"
checkpoint="logs_gpu/fixed-tetris-lr-1e-5/PPO/rl_model_3800000_steps.zip"

action_type = 'grouped'
env_max_step = 18000
reward_type='standard'
reward_scaling=None
board_size = (10,10)
env = TetrisEnv(board_size=board_size, action_type=action_type, reward_type=reward_type, max_steps=env_max_step, reward_scaling=reward_scaling)

# loading model
# model = PPO.load(checkpoint,force_reset=False)
#learnt hyperparameters
lr = 1e-5
# lr = 4.579846751190279e-05 # default 1e05
gae_lambda = 0.95
# gae_lambda = 0.8954544738495033 # default 0.95
max_grad_norm = 0.5
# max_grad_norm = 0.6600052035605939 # default 0.5
ent_coef = 0.0
# ent_coef = 0.08232010781605359 # default 0.0
vf_coef = 0.5
# vf_coef = 0.5183386093886262 # default 0.5
n_steps = 2048
# n_steps = 37
# set hyperparameters
batch_size = 512
# model and callbacks
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=folder, device="cuda:0",
    batch_size=batch_size,
    learning_rate=lr, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, n_steps=n_steps
)
model.set_parameters(checkpoint)
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
