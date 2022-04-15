from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
import argparse
from stable_baselines3.common.vec_env import VecFrameStack

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# environment arguments
parser.add_argument('--env-size', default='20,10', help='size of the board')
parser.add_argument('--env-action-type', default='grouped', help='type of the action to take [grouped, semigrouped, standard]')
parser.add_argument('--env-output', default='image', help='what the env output is')
parser.add_argument('--env-score-reward', type=bool, default=True, help='whether the output should be score instead of just lines cleared')
parser.add_argument('--env-reward-scale',type=float, default=1.0, help='Should the reward be scaled')
# learning arguments
parser.add_argument('--logdir', default='logs/test/', help='where should stuff be logged')
parser.add_argument('--device', default='cuda:0', help='What device should the model be trained on')
parser.add_argument('--save-freq', type=int, default=100000, help='how frequently should the model be checkpointed')
parser.add_argument('--eval-freq', type=int, default=2000, help='how frequently should the model be evaluated')
parser.add_argument('--logg-freq', type=int, default=1, help='how frequently should the model be logged')
parser.add_argument('--eval-num', type=int, default=10, help='how many episodes per model evaluation')
parser.add_argument('--training-num', type=int, default=1000000, help='how many episodes should the model be trained for')

args = parser.parse_args()
model_type = "PPO"
model_folder = args.logdir + model_type + "/"
board_size = tuple([int(i) for i in args.env_size.split(',')])
env_max_step = 18000
reward_type='standard'
reward_scaling=None

def make_env(board_size, args, seed=0):
    def _init():
        env = TetrisEnv(board_size=board_size, action_type=args.env_action_type, reward_type=reward_type, max_steps=env_max_step, reward_scaling=reward_scaling)
        return env
    return _init

def convert(steps, num_cpu):
        return max(steps // num_cpu, 1)

# environments
num_cpu = 1
env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])
eval_env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])

save_freq = convert(args.save_freq, num_cpu)
eval_freq = convert(args.eval_freq, num_cpu)
logg_freq = convert(args.logg_freq, num_cpu)
training_num = convert(args.training_num, num_cpu)
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
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=args.logdir, device=args.device,
    batch_size=batch_size,
    learning_rate=lr, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, n_steps=n_steps
)
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_folder)
eval_callback = EvalCallback(eval_env, best_model_save_path=model_folder,
                             log_path=model_folder, eval_freq=eval_freq,
                             deterministic=False, render=False, n_eval_episodes=args.eval_num)

# trainig
model.learn(total_timesteps=training_num, log_interval=logg_freq, callback=[checkpoint_callback, eval_callback])
model.save(model_folder + "/final_model")
