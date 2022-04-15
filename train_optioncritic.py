from stable_baselines3.common.env_checker import check_env
from option_critic.option_critic import OptionCritic, EvalCallbackOptionCritic, CheckpointCallbackOptionCritic
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
import argparse
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# environment arguments
parser.add_argument('--env-size', default='20,10', help='size of the board')
parser.add_argument('--env-simplified', type=bool, default=False, help='simplified version of the game')
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
model_type = "OptionCritic"
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

# environments
num_cpu = 1
env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])
eval_env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])

#learnt hyperparameters
lr = 5.2715943476112373e-05
update_frequency = 12 # default was 4
freeze_interval = 1356
entropy_reg = 1.5374056773600333e-05
termination_reg = 1.307299921895168e-05
buffer_size = 22549 # default 16000
epsilon_decay = 71712.5903207788
# set hyperparameters
batch_size = 512
num_options = 8
learning_starts = 2048
# model and callbacks
model = OptionCritic(env, logdir=args.logdir, device=args.device,
    num_options=num_options, learning_starts=learning_starts,
    lr=lr, update_frequency=update_frequency, freeze_interval=freeze_interval, entropy_reg=entropy_reg, termination_reg=termination_reg, buffer_size=buffer_size, epsilon_decay=epsilon_decay
)
checkpoint_callback = CheckpointCallbackOptionCritic(freq=args.save_freq, save_path=model_folder)
eval_callback = EvalCallbackOptionCritic(eval_env, best_model_save_path=model_folder,
                             log_path=model_folder, freq=args.eval_freq,
                             deterministic=False, n_eval_episodes=args.eval_num)

# trainig
model.learn(total_timesteps=args.training_num, log_interval=args.logg_freq, callback=[checkpoint_callback, eval_callback])
model.save(model_folder + "/final_model")
