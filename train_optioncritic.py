from stable_baselines3.common.env_checker import check_env
from option_critic.option_critic import OptionCritic, EvalCallbackOptionCritic, CheckpointCallbackOptionCritic
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
parser.add_argument('--logg-freq', type=int, default=8, help='how frequently should the model be logged')
parser.add_argument('--eval-num', type=int, default=50, help='how many episodes per model evaluation')
parser.add_argument('--training-num', type=int, default=1000000, help='how many episodes should the model be trained for')
# custom model arguments


args = parser.parse_args()
model_type = "OptionCritic"
model_folder = args.logdir + model_type + "/"
board_size = tuple([int(i) for i in args.env_size.split(',')])

# environments
env = TetrisEnv(board_size=board_size, action_type=args.env_action_type, output_type=args.env_output, simplified=args.env_simplified, score_reward=args.env_score_reward, scale_reward=args.env_reward_scale, max_steps=args.env_max_step)
eval_env = Monitor(TetrisEnv(board_size=board_size, action_type=args.env_action_type, output_type=args.env_output, simplified=args.env_simplified, score_reward=args.env_score_reward, scale_reward=args.env_reward_scale, max_steps=args.env_max_step))

# model and callbacks
model = OptionCritic(env, update_frequency=1, logdir=args.logdir, num_options=8, device=args.device)
checkpoint_callback = CheckpointCallbackOptionCritic(freq=args.save_freq, save_path=model_folder)
eval_callback = EvalCallbackOptionCritic(eval_env, best_model_save_path=model_folder,
                             log_path=model_folder, freq=args.eval_freq,
                             deterministic=False, n_eval_episodes=args.eval_num)

# trainig
model.learn(total_timesteps=args.training_num, log_interval=args.logg_freq, callback=[checkpoint_callback, eval_callback])
model.save(folder + type(model).__name__ + "/final_model")
