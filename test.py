import gym

import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
from option_critic.option_critic import OptionCritic

def create_PPO_Model(logdir, device):
    lr = 1e-5 # HPO 4.579846751190279e-05
    gae_lambda = 0.95 # HPO 0.8954544738495033
    max_grad_norm = 0.5 # HPO 0.6600052035605939
    ent_coef = 0.0 # HPO 0.08232010781605359
    vf_coef = 0.5 # HPO 0.5183386093886262
    n_steps = 2048 # HPO 37
    batch_size = 512
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir, device=device,
        batch_size=batch_size,
        learning_rate=lr, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, n_steps=n_steps
    )
    return model

def create_DQN_Model(logdir, device):
    lr = 0.00048530177474105597 #default was 1e-5
    tau = 0.6894955031710317
    train_freq = 9 # default was 4
    target_update_interval = 7863
    exploration_fraction = 0.8069896792007639
    buffer_size = 100#hpo 38846
    # set hyperparameters
    batch_size = 512
    learning_starts = 2048
    model = DQN("CnnPolicy", env , verbose=1, tensorboard_log=logdir, device=device,
        batch_size=batch_size, learning_starts=learning_starts,
        learning_rate=lr, tau=tau, train_freq=train_freq, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction, buffer_size=buffer_size
    )
    return model

def create_Option_Model(logdir, device, num_options):
    lr = 5.2715943476112373e-05
    update_frequency = 12 # default was 4
    freeze_interval = 1356
    entropy_reg = 0.01#1.5374056773600333e-05
    termination_reg = 0.01#1.307299921895168e-05
    buffer_size = 22549 # default 16000
    epsilon_decay = 71712.5903207788
    # set hyperparameters
    batch_size = 512
    learning_starts = 2048
    # model and callbacks
    model = OptionCritic(env, logdir=logdir, device=device,
        num_options=num_options, learning_starts=learning_starts,
        lr=lr, update_frequency=update_frequency, freeze_interval=freeze_interval, entropy_reg=entropy_reg, termination_reg=termination_reg, buffer_size=buffer_size, epsilon_decay=epsilon_decay
    )
    return model

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# environment arguments
parser.add_argument('--env-size', default='10,10', help='size of the board')
parser.add_argument('--env-action-type', default='grouped', help='type of the action to take [grouped, semigrouped, standard]')
parser.add_argument('--env-reward-type', default='standard', help='what the reward type should be [standard, no piece drop, only lines, num lines]')
parser.add_argument('--env-reward-scaling', default=None, help='Should the reward be scaled [multi, log]')
parser.add_argument('--env-max-steps', default=18000, help='Maximum steps in environment (to prevent infinite runs)')
#
parser.add_argument('--logdir', default='logs/test/', help='where should stuff be logged')
parser.add_argument('--model', default='best_model', help='name of the checkpoint')
parser.add_argument('--eval-num', type=int, default=1000, help='number of episodes to evaluate')
parser.add_argument('--model-type', default="PPO", help='what model to be using [PPO, DQN, Option]')
parser.add_argument('--options', type=int, default=8, help='how many options')

args = parser.parse_args()

folder = args.logdir
checkpoint = args.logdir + args.model
board_size = (10,10)
board_size = tuple([int(i) for i in args.env_size.split(',')])
env = TetrisEnv(board_size=board_size, action_type=args.env_action_type, reward_type=args.env_reward_type, max_steps=args.env_max_steps, reward_scaling=args.env_reward_scaling)

# loading model
# model = PPO.load(checkpoint,force_reset=False)
print(args.model_type)
if args.model_type == "PPO":
    model = create_PPO_Model(folder, "cuda:0")
    model.set_parameters(checkpoint)
elif args.model_type == "DQN":
    model = create_DQN_Model(folder, "cuda:0")
    model.set_parameters(checkpoint)
elif args.model_type == "Option":
    model = create_Option_Model(folder, "cuda:0", args.options)
    model.load(checkpoint)
else:
    raise ValueError('Model type'+ args.model+ 'not recognized.')


# average steps - how long the model survive
# average reward - how much reward is the models obtaining
# average lines cleared - how many lines are the models clearing
# average blocks placed
# percentage of actions result in a block being placed
# distribution of lines cleared
# distribution of actions taken
# distribution of index where the block is dropped
# level distribution
# combo distribution

average_steps = 0
average_reward = 0
average_lines_cleared = 0
average_blocks_placed = 0
distribution_lines_cleared = [0] * 5 # only possible to clear between 0 and 4 lines
distribution_actions_taken = [0] * len(env._action_set)
distribution_level = [0] * 10
distribution_combo = [0] * 11
distribution_dropped_x = [0] * env.state.X_BOARD
distribution_dropped_y = [0] * env.state.Y_BOARD
for _ in range(args.eval_num):
    game_over = False; obs = env.reset(); current_option = None
    while not game_over:
        if args.model_type == "Option":
            action, current_option = model.predict(obs, deterministic=False, current_option=current_option) # option critic version
        else:
            action, _ = model.predict(obs, deterministic=False)
        obs, reward, game_over, info = env.step(action)
        average_steps += 1
        average_reward += reward
        average_lines_cleared += info['lines cleared']
        average_blocks_placed = average_blocks_placed + 1 if info['block placed'] is not None else average_blocks_placed
        distribution_lines_cleared[info['lines cleared']] += 1
        distribution_actions_taken[action] += 1
        distribution_level[info['level']] += 1
        if info['combo'] == -1:
            distribution_combo[0] += 1
        else:
            distribution_combo[info['combo']+1] += 1
        if info['block placed'] is not None:
            distribution_dropped_x[info['block placed'][0]] += 1
            distribution_dropped_y[info['block placed'][1]] += 1
average_steps /= args.eval_num
average_reward /= args.eval_num
average_lines_cleared /= args.eval_num
average_blocks_placed /= args.eval_num
percentage_steps_where_blocks_placed = average_blocks_placed / average_steps
print(f"Per episode, average steps: {average_steps}, average reward: {average_reward}, average lines cleared: {average_lines_cleared}, percentage steps where blocks placed: {percentage_steps_where_blocks_placed}")
print("Lines cleared distribution", ",".join([str(x) for x in distribution_lines_cleared]))
print("Actions taken distribution", ",".join([str(x) for x in distribution_actions_taken]))
print("Level distribution", ",".join([str(x) for x in distribution_level]))
print("Combo distribution", ",".join([str(x) for x in distribution_combo]))
print("Blocks dropped at x distribution", ",".join([str(x) for x in distribution_dropped_x]))
print("Blocks dropped at y distribution", ",".join([str(x) for x in distribution_dropped_y]))


# game_over      = False
# obs            = env.reset()
# current_option = None
# # env.measure_step_time(verbose=True)
# env.render(wait_sec=1, mode='image', verbose=True)
# while not game_over:
#     action, _ = model.predict(obs, deterministic=False)
#     print("Action", action)
#     # action, current_option = model.predict(obs, deterministic=False, current_option=current_option) # option critic version
#     # print(f"Action: {action}| option: {current_option}")
#     obs, reward, game_over, info = env.step(action)
#
#     env.render(wait_sec=1, mode='image', verbose=True)
#     print("Reward", reward)
