import argparse
import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris import TetrisEnv
from option_critic import OptionCritic, EvalCallbackOptionCritic, CheckpointCallbackOptionCritic

def init_arg_parse():
    parser = argparse.ArgumentParser(description="Option Critic PyTorch")
    # environment arguments
    parser.add_argument('--env-size', default='20,10', help='size of the board')
    parser.add_argument('--env-simplified', type=bool, default=False, help='simplified version of the game')
    parser.add_argument('--env-action-type', default='grouped', help='type of the action to take [grouped, semigrouped, standard]')
    parser.add_argument('--env-output', default='image', help='what the env output is')
    parser.add_argument('--env-score-reward', type=bool, default=True, help='whether the output should be score instead of just lines cleared')
    parser.add_argument('--env-reward-scale',type=float, default=1.0, help='Should the reward be scaled')
    parser.add_argument('--env-vectorized-num', default=None, help='should we vectorize the environment and how much')
    parser.add_argument('--env-max-step', type=int, default=1000, help='Maximum steps in the environment')
    # learning arguments
    parser.add_argument('--logdir', default='logs/test/', help='where should stuff be logged')
    parser.add_argument('--device', default='cuda:0', help='What device should the model be trained on')
    parser.add_argument('--save-freq', type=int, default=100000, help='how frequently should the model be checkpointed')
    parser.add_argument('--eval-freq', type=int, default=2000, help='how frequently should the model be evaluated')
    parser.add_argument('--logg-freq', type=int, default=500, help='how frequently should the model be logged')
    parser.add_argument('--eval-num', type=int, default=50, help='how many episodes per model evaluation')
    parser.add_argument('--training-num', type=int, default=1000000, help='how many episodes should the model be trained for')
    return parser

def get_model(model_type, args):
    def make_env(board_size, action_type, output_type, simplified, score_reward, scale_reward, seed=0):
        def _init():
            env = TetrisEnv(board_size=board_size, action_type=action_type, output_type=output_type, simplified=simplified, score_reward=score_reward, scale_reward=scale_reward)
            return env
        return _init

    def convert(steps, num_cpu):
        return max(steps // num_cpu, 1)

    # env args
    board_size = tuple([int(i) for i in args.env_size.split(',')])
    simplified = args.env_simplified
    action_type = args.env_action_type
    output_type = args.env_output
    score_reward = args.env_score_reward
    scale_reward = args.env_reward_scale
    num_cpu = args.env_vectorized_num if args.env_vectorized_num is not None else 1
    max_step = args.env_max_step

    if num_cpu > 1:
        env = DummyVecEnv([make_env(board_size, action_type=action_type, output_type=output_type, simplified=simplified, score_reward=score_reward, scale_reward=scale_reward) for i in range(num_cpu)])
    else:
        env = TetrisEnv(board_size=board_size, action_type=action_type, output_type=output_type, simplified=simplified, score_reward=score_reward, scale_reward=scale_reward)
    eval_env = Monitor(TetrisEnv(board_size=board_size, action_type=action_type, output_type=output_type, simplified=simplified, score_reward=score_reward, scale_reward=scale_reward))

    # learning args
    folder = args.logdir
    device = args.device
    save_freq = args.save_freq
    eval_freq = args.eval_freq
    logging_freq = args.logg_freq
    eval_num = args.eval_num
    num_steps = args.training_num

    model = {'PPO':PPO, 'DQN':DQN, 'OptionCritic':OptionCritic}[model_type]
    model_folder = folder + type(model).__name__ + "/"

    # callbacks
    if model_type != 'OptionCritic':
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_folder)
        eval_callback = EvalCallback(eval_env, best_model_save_path=model_folder,
                                     log_path=model_folder, eval_freq=eval_freq,
                                     deterministic=False, render=False, n_eval_episodes=100)
    else:
        checkpoint_callback = CheckpointCallbackOptionCritic(freq=5000, save_path=model_folder)
        eval_callback = EvalCallbackOptionCritic(eval_env, best_model_save_path=model_folder,
                                     log_path=model_folder, freq=eval_freq,
                                     deterministic=False, n_eval_episodes=100, max_steps_ep=18000)

    return env, model, folder, [eval_callback, checkpoint_callback], device, num_steps, logging_freq
