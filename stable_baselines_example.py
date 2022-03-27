import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris import TetrisEnv
from option_critic import OptionCritic

def make_env(seed=0):
    def _init():
        env = TetrisEnv(board_size=(10,10), grouped_actions=True, only_squares=True, no_rotations=True)
        return env
    return _init

def convert(steps, num_cpu):
    return max(steps // num_cpu, 1)

num_cpu = 4
num_steps = 10000000 # 10 million steps
save_freq = convert(num_steps/10, num_cpu)
eval_freq = convert(2048, num_cpu)
folder = "./logs/PPO-tetris_with_vectorized_env"

# env = gym.make("Breakout-v0")
env = DummyVecEnv([make_env() for i in range(num_cpu)])
# env = Monitor(TetrisEnv(board_size=(10,10), grouped_actions=True, only_squares=True, no_rotations=True))
eval_env = Monitor(TetrisEnv(board_size=(10,10), grouped_actions=True, only_squares=True, no_rotations=True))

checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=folder, name_prefix='PPO_model')
eval_callback = EvalCallback(eval_env, best_model_save_path=folder,
                             log_path=folder, eval_freq=eval_freq,
                             deterministic=False, render=False, n_eval_episodes=100)

# model = DQN("CnnPolicy", env, verbose=0,buffer_size=10000, tensorboard_log='logs/DQN-tetris')
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=folder)
# model = OptionCritic(env, tensorboard_log='logs/OptionCritic_2')

model = model.learn(total_timesteps=num_steps, log_interval=1, callback=[eval_callback, checkpoint_callback])
model.save(folder + "/final_model")

# game_over = False
# obs = env.reset()
# while not game_over:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, game_over, info = env.step(action)
#     env.render()
#     print("Game over:", game_over)
