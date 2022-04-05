import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris import TetrisEnv
from option_critic import OptionCritic, EvalCallbackOptionCritic, CheckpointCallbackOptionCritic

def make_env(board_size, action_type, output_type, simplified, score_reward, scale_reward, seed=0):
    def _init():
        env = TetrisEnv(board_size=board_size, action_type=action_type, output_type=output_type, simplified=simplified, score_reward=score_reward, scale_reward=scale_reward)
        return env
    return _init

def convert(steps, num_cpu):
    return max(steps // num_cpu, 1)

num_cpu = 1
num_steps = 1000000 # 2 million steps
save_freq = convert(num_steps/10, num_cpu)
eval_freq = convert(2000, num_cpu)
logging_freq = convert(500, num_cpu)
folder = "./logs/tetris_6x6_simplified_1.0_score/"

board_size = (6,6)
action_type=['grouped', 'semigrouped', 'standard'][0]
output_type='image'
simplified=True
score_reward=True
scale_reward=1.0

# fixed board rendering not removing rows
# lowered learning rate from 0.0003 to 0.00003
# large arnge of possible rewards?

# env = gym.make("Breakout-v0")
if num_cpu > 1:
    env = DummyVecEnv([make_env(board_size) for i in range(num_cpu)])
else:
    env = TetrisEnv(board_size=board_size, action_type=action_type, output_type=output_type, simplified=simplified, score_reward=score_reward, scale_reward=scale_reward)
eval_env = Monitor(TetrisEnv(board_size=board_size, action_type=action_type, output_type=output_type, simplified=simplified, score_reward=score_reward, scale_reward=scale_reward))

# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=folder, learning_rate=1e-5)
model = DQN("CnnPolicy", env, verbose=0, buffer_size=10000, tensorboard_log=folder)
# model = OptionCritic(env, update_frequency=1, logdir=folder, num_options=8)
model_folder = folder + type(model).__name__ + "/"

# loading model
# model = DQN.load(model_folder + "_model_200000_steps")

# callbacks
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_folder)
# checkpoint_callback = CheckpointCallbackOptionCritic(freq=5000, save_path=model_folder)
eval_callback = EvalCallback(eval_env, best_model_save_path=model_folder,
                             log_path=model_folder, eval_freq=eval_freq,
                             deterministic=False, render=False, n_eval_episodes=100)
# eval_callback = EvalCallbackOptionCritic(eval_env, best_model_save_path=model_folder,
#                              log_path=model_folder, freq=eval_freq,
#                              deterministic=False, n_eval_episodes=100, max_steps_ep=18000)

# trainig
model = model.learn(total_timesteps=num_steps, log_interval=logging_freq, callback=[eval_callback, checkpoint_callback])
# model.learn(max_steps_total=num_steps, max_steps_ep=18000, callback=[eval_callback, checkpoint_callback]) # for the option critic
# model.save(model_folder + "final_model")
