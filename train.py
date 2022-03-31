import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris import TetrisEnv
from option_critic import OptionCritic

def make_env(board_size, seed=0):
    def _init():
        env = TetrisEnv(board_size=board_size, grouped_actions=True, only_squares=True, no_rotations=True)
        return env
    return _init

def convert(steps, num_cpu):
    return max(steps // num_cpu, 1)

num_cpu = 4
num_steps = 2000000 # 2 million steps
save_freq = convert(num_steps/10, num_cpu)
eval_freq = convert(2000, num_cpu)
size = (6,6)

# fixed board rendering not removing rows
# lowered learning rate from 0.0003 to 0.00003
# large arnge of possible rewards?

model_type = "ActionCritic"
folder = "./logs/fixing_option_critic7/"
model_folder = folder + model_type + "/"

# env = gym.make("Breakout-v0")
# env = DummyVecEnv([make_env(size) for i in range(num_cpu)])
env = TetrisEnv(board_size=size, grouped_actions=True, only_squares=True, no_rotations=True)
eval_env = Monitor(TetrisEnv(board_size=size, grouped_actions=True, only_squares=True, no_rotations=True))

# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=folder, learning_rate=1e-5)
# model = DQN("CnnPolicy", env, verbose=0,buffer_size=10000, tensorboard_log=folder)
model = OptionCritic(env)

# loading model
# model = DQN.load(model_folder + "_model_200000_steps")

# callbacks
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=folder, name_prefix=model_folder + '_model')
eval_callback = EvalCallback(eval_env, best_model_save_path=model_folder,
                             log_path=model_folder, eval_freq=eval_freq,
                             deterministic=False, render=False, n_eval_episodes=100)

# trainig
# model = model.learn(total_timesteps=num_steps, log_interval=1)#, callback=[eval_callback, checkpoint_callback])
model.learn(max_steps_total=1000000, max_steps_ep=18000) # for the option critic
# model.save(model_folder + "final_model")
