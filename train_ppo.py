from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
import argparse

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# environment arguments
parser.add_argument('--env-size', default='20,10', help='size of the board')
parser.add_argument('--env-simplified', type=bool, default=False, help='simplified version of the game')
parser.add_argument('--env-action-type', default='grouped', help='type of the action to take [grouped, semigrouped, standard]')
parser.add_argument('--env-output', default='image', help='what the env output is')
parser.add_argument('--env-score-reward', type=bool, default=True, help='whether the output should be score instead of just lines cleared')
parser.add_argument('--env-reward-scale',type=float, default=1.0, help='Should the reward be scaled')
parser.add_argument('--env-max-step', type=int, default=1000, help='Maximum steps in the environment')
# learning arguments
parser.add_argument('--logdir', default='logs/test/', help='where should stuff be logged')
parser.add_argument('--device', default='cuda:0', help='What device should the model be trained on')
parser.add_argument('--save-freq', type=int, default=100000, help='how frequently should the model be checkpointed')
parser.add_argument('--eval-freq', type=int, default=2000, help='how frequently should the model be evaluated')
parser.add_argument('--logg-freq', type=int, default=1, help='how frequently should the model be logged')
parser.add_argument('--eval-num', type=int, default=10, help='how many episodes per model evaluation')
parser.add_argument('--training-num', type=int, default=1000000, help='how many episodes should the model be trained for')
# hyperparmaeters
parser.add_argument('--lr', type=int, default=1e-5)
# gamma, gae_lambda, clip_range, normalize_advantage,target_kl
# run optimization
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--n-steps', type=int, default=2048)
#n_steps


args = parser.parse_args()
num_cpu = 1
model_type = "PPO"
model_folder = args.logdir + model_type + "/"
board_size = tuple([int(i) for i in args.env_size.split(',')])

def convert(steps, num_cpu):
        return max(steps // num_cpu, 1)

save_freq = convert(args.save_freq, num_cpu)
eval_freq = convert(args.eval_freq, num_cpu)
logg_freq = convert(args.logg_freq, num_cpu)
training_num = convert(args.training_num, num_cpu)

def make_env(board_size, args, seed=0):
    def _init():
        env = TetrisEnv(board_size=board_size, action_type=args.env_action_type, output_type=args.env_output, simplified=args.env_simplified, score_reward=args.env_score_reward, scale_reward=args.env_reward_scale, max_steps=args.env_max_step)
        return env
    return _init

# environments
env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])
eval_env = Monitor(TetrisEnv(board_size=board_size, action_type=args.env_action_type, output_type=args.env_output, simplified=args.env_simplified, score_reward=args.env_score_reward, scale_reward=args.env_reward_scale, max_steps=args.env_max_step))

# model and callbacks
model = PPO("CnnPolicy", env,
    learning_rate=args.lr, batch_size=args.batch_size, n_steps=args.n_steps,
    verbose=1, tensorboard_log=args.logdir, device=args.device)
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_folder)
eval_callback = EvalCallback(eval_env, best_model_save_path=model_folder,
                             log_path=model_folder, eval_freq=eval_freq,
                             deterministic=False, render=False, n_eval_episodes=args.eval_num)

# trainig
model.learn(total_timesteps=training_num, log_interval=logg_freq, callback=[checkpoint_callback, eval_callback])
model.save(folder + type(model).__name__ + "/final_model")
