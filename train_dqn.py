from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
import argparse
from stable_baselines3.common.vec_env import VecFrameStack

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# environment arguments
parser.add_argument('--env-size', default='20,10', help='size of the board')
parser.add_argument('--env-action-type', default='grouped', help='type of the action to take [grouped, semigrouped, standard]')
parser.add_argument('--env-reward-type', default='standard', help='what the reward type should be [standard, no piece drop, only lines, num lines]')
parser.add_argument('--env-reward-scaling', default=None, help='Should the reward be scaled [multi, log]')
parser.add_argument('--env-max-step', default=18000, help='Maximum steps in environment (to prevent infinite runs)')
# learning arguments
parser.add_argument('--logdir', default='logs/test/', help='where should stuff be logged')
parser.add_argument('--device', default='cuda:0', help='What device should the model be trained on')
parser.add_argument('--save-freq', type=int, default=100000, help='how frequently should the model be checkpointed')
parser.add_argument('--eval-freq', type=int, default=2000, help='how frequently should the model be evaluated')
parser.add_argument('--logg-freq', type=int, default=1, help='how frequently should the model be logged')
parser.add_argument('--eval-num', type=int, default=10, help='how many episodes per model evaluation')
parser.add_argument('--training-num', type=int, default=1000000, help='how many episodes should the model be trained for')

args = parser.parse_args()
model_type = "DQN"
model_folder = args.logdir + model_type + "/"
board_size = tuple([int(i) for i in args.env_size.split(',')])
env_max_step = 18000
reward_type='standard'
reward_scaling=None

def make_env(board_size, args, seed=0):
    def _init():
        env = TetrisEnv(board_size=board_size, action_type=args.env_action_type, reward_type=args.env_reward_type, max_steps=args.env_max_steps, reward_scaling=args.env_reward_scaling)
        return env
    return _init

# environments
num_cpu = 1
env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])
eval_env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])
print(env.observation_space.shape)

#learnt hyperparameters
lr = 0.00048530177474105597 #default was 1e-5
tau = 0.6894955031710317
train_freq = 9 # default was 4
target_update_interval = 7863
exploration_fraction = 0.8069896792007639
buffer_size = 100#hpo 38846
# set hyperparameters
batch_size = 512
learning_starts = 2048
# model and callbacks
model = DQN(
    "CnnPolicy", env , verbose=1, tensorboard_log=args.logdir, device=args.device,
    batch_size=batch_size, learning_starts=learning_starts,
    learning_rate=lr, tau=tau, train_freq=train_freq, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction, buffer_size=buffer_size
)
checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=model_folder)
eval_callback = EvalCallback(eval_env, best_model_save_path=model_folder,
                             log_path=model_folder, eval_freq=args.eval_freq,
                             deterministic=False, render=False, n_eval_episodes=args.eval_num)

# trainig
steps = int(args.training_num / 40000)
for step in range(steps):
    if step == 0:
        model.learn(total_timesteps=40000, reset_num_timesteps=True, log_interval=args.logg_freq, callback=[checkpoint_callback, eval_callback]) #40000
    else:
        model.learn(total_timesteps=40000, reset_num_timesteps=False, log_interval=args.logg_freq, callback=[checkpoint_callback, eval_callback]) #40000
model.save(model_folder + "/final_model")
