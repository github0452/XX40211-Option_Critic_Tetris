import gym
import random
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tetris.tetris import TetrisEnv
from option_critic.option_critic import OptionCritic
from tqdm import tqdm
import numpy as np
import os

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
    buffer_size = 100 #38846 - reduced for testing as it wasn't important
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
    buffer_size = 22549 # default 16000
    epsilon_decay = 71712.5903207788
    # set hyperparameters
    batch_size = 512
    learning_starts = 2048
    entropy_reg = 0.01
    termination_reg = 0.01
    # model and callbacks
    model = OptionCritic(env, logdir=logdir, device=device,
        num_options=num_options, learning_starts=learning_starts,
        lr=lr, update_frequency=update_frequency, freeze_interval=freeze_interval, entropy_reg=entropy_reg, termination_reg=termination_reg, buffer_size=buffer_size, epsilon_decay=epsilon_decay
    )
    return model

# loading model
def load_model(model_type, checkpoint, num_options=8):
    # model = PPO.load(checkpoint,force_reset=False)
    if model_type == "PPO":
        model = create_PPO_Model(folder, "cuda:0")
        model.set_parameters(checkpoint)
    elif model_type == "DQN":
        model = create_DQN_Model(folder, "cuda:0")
        model.set_parameters(checkpoint)
    elif model_type == "Option":
        model = create_Option_Model(folder, "cuda:0", num_options)
        model.load(checkpoint)
    elif model_type == "random":
        model = None
    else:
        raise ValueError('Model type'+ model_type+ 'not recognized.')
    return model

# from https://stackoverflow.com/questions/10461531/merge-and-sum-of-two-dictionaries
def reducer(accumulator, element):
    for key, value in element.items():
        accumulator[key] = accumulator.get(key, 0) + value
    return accumulator

from collections import defaultdict

def get_stats(args, model, model_type, num_options):
    lines_cleared = np.zeros((5)); pieces_placed = 0
    piece_dict = {"I":0, "J":1, "S":2, "Z":3, "L":4, "O":5, "T":6}
    action_dist_per_piece = np.zeros((7, len(env._action_set)))
    num_blocks_filled = np.zeros((env.state.Y_BOARD, env.state.X_BOARD))
    when_blocks_filled = np.zeros((env.state.Y_BOARD, env.state.X_BOARD))
    piece_dist_options = np.zeros((num_options, env.state.Y_BOARD, env.state.X_BOARD))
    score_types = ['Lines', 'Combo', 'softdrop', 'harddrop']
    score_breakdown = np.zeros((4))
    score_max = 0; score_min = 100000
    if model_type == "Option":
        option_lengths = {opt:[] for opt in range(num_options)}
        curr_op_len = 0
        dist_piece_per_option = np.zeros((num_options, 7))
        dist_actions_per_option = np.zeros((num_options, len(env._action_set)))
        dist_blocks_filled_per_option = np.zeros((num_options, env.state.Y_BOARD, env.state.X_BOARD))
    for i in tqdm(range(args.eval_num)):
        game_over = False; obs = env.reset(); current_option = None; curr_step = 0; eps_reward = 0
        board_counter = np.zeros((env.state.Y_BOARD, env.state.X_BOARD))
        board_counter_options = np.zeros((num_options, env.state.Y_BOARD, env.state.X_BOARD))
        while not game_over:
            # getting info about the state
            piece = env.state.curr.type
            # selecting action
            if model_type == "Option":
                prev_option = current_option
                action, current_option = model.predict(obs, deterministic=False, current_option=current_option) # option critic version
            elif model_type == "random":
                action = random.choice(env._action_set)
            else:
                action, _ = model.predict(obs, deterministic=False)
            # taking action
            obs, reward, game_over, info = env.step(action)
            # update stats
            eps_reward += reward
            for i in range(len(score_types)):
                score_breakdown[i] += info[score_types[i]]
            lines_cleared[info['lines cleared']] += 1
            action_dist_per_piece[piece_dict[piece]][action] += 1
            board_counter += env.state.mini_board
            pieces_placed += 1 if info['block placed'] is not None else 0
            curr_step += 1
            if model_type == "Option":
                if prev_option is None:
                    pass
                elif prev_option == current_option: # as in the option does not terminate
                    curr_op_len += 1
                else:
                    option_lengths[prev_option].append(curr_op_len)
                    curr_op_len = 1
                dist_piece_per_option[current_option][piece_dict[piece]] += 1
                dist_actions_per_option[current_option][action] += 1
                if info['block placed'] is not None:
                    piece_dist_options[current_option][info['block placed'][1], info['block placed'][0]] += 1
        when_blocks_filled[board_counter > 0] += curr_step - board_counter[board_counter > 0]
        num_blocks_filled += env.state.mini_board
        score_max = max(score_max, eps_reward)
        score_min = min(score_min, eps_reward)
    when_blocks_filled[num_blocks_filled > 0] = np.divide(when_blocks_filled[num_blocks_filled > 0], num_blocks_filled[num_blocks_filled>0])
    when_blocks_filled[num_blocks_filled == 0] = -1
    # episode stats
    print("Averaged per episode")
    average_actions = np.sum(action_dist_per_piece)/args.eval_num
    average_rewards = np.sum(score_breakdown)/args.eval_num
    average_lines = sum(lines_cleared[1:])/args.eval_num
    block_placed_perc = pieces_placed/np.sum(action_dist_per_piece)
    score_lines, score_combo, score_softdrop, score_harddrop = tuple(score_breakdown)
    action_0lines,action_1lines,action_2lines,action_3lines,action_4lines  = tuple(lines_cleared.tolist())
    # general game stats
    print("Actions,Rewards,Max_reward,Min_reward,Lines_cleared,Blocked_placed_%,\
                score_lines,score_combo,score_softdrop,score_harddrop,\
                action_0lines,action_1lines,action_2lines,action_3lines,action_4lines")
    print(f"{average_actions},{average_rewards},{score_max},{score_min},{average_lines},{block_placed_perc},\
            {score_lines},{score_combo},{score_softdrop},{score_harddrop},\
            {action_0lines},{action_1lines},{action_2lines},{action_3lines},{action_4lines}")
    print(f"Actions,{','.join([str(x) for x in range(40)])}")
    print(f"Number,{','.join(np.sum(action_dist_per_piece, axis=0).astype(str).tolist())}")
    print("Tetris board: how frequently is a block filled")
    for row in num_blocks_filled/args.eval_num:
        print(','.join(row.astype(str).tolist()))
    print("Tetris board: when is a block filled on average")
    for row in when_blocks_filled:
        print(','.join(row.astype(str).tolist()))
    # piece stats
    for piece, i in piece_dict.items():
        print(f"{piece},{','.join(action_dist_per_piece[i].astype(str).tolist())}")
    # option stats
    if model_type == "Option":
        option_lengths = [option_lengths[x] for x in range(num_options)]
        average_option_active = [sum(x) for x in option_lengths]
        average_option_length = [sum(x)/len(x) for x in option_lengths]
        print(f"Option_active_distribution,{','.join([str(x) for x in average_option_active])}")
        print(f"Average_option_lengths_distribution,{','.join([str(x) for x in average_option_length])}")
        for i in range(num_options):
            print(f"Piece_distribution for option {i}: {dist_piece_per_option[i].tolist()}")
            print(f"Actions_taken_distribution for option {i}: {dist_actions_per_option[i].tolist()}")
            # print(f"Block filled distribution for option {i}:")
            # for row in dist_blocks_filled_per_option[i]:
            #     print(row)
            print("end")

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# environment arguments
parser.add_argument('--env-size', default='10,10', help='size of the board')
parser.add_argument('--env-action-type', default='grouped', help='type of the action to take [grouped, semigrouped, standard]')
parser.add_argument('--env-reward-type', default='standard', help='what the reward type should be [standard, no piece drop, only lines, num lines]')
parser.add_argument('--env-reward-scaling', default=None, help='Should the reward be scaled [multi, log]')
parser.add_argument('--env-max-steps', default=18000, help='Maximum steps in environment (to prevent infinite runs)')
#
parser.add_argument('--logdir', default='logs/test/', help='where should stuff be logged')
parser.add_argument('--model', default=None, help='name of the checkpoint')
parser.add_argument('--eval-num', type=int, default=100, help='number of episodes to evaluate')
parser.add_argument('--model-type', default=None, help='what model to be using [PPO, DQN, Option]')
parser.add_argument('--options', type=int, default=8, help='how many options')

args = parser.parse_args()
# checkpoint = args.logdir + args.model
board_size = tuple([int(i) for i in args.env_size.split(',')])
env = TetrisEnv(board_size=board_size, action_type=args.env_action_type, reward_type=args.env_reward_type, max_steps=args.env_max_steps, reward_scaling=args.env_reward_scaling)

if args.model_type is not None:
    iteration = [(args.model_type, args.option, args.logdir)]
else:
    iteration = [('random', 1, args.logdir), ('PPO', 1, args.logdir+'PPO/'), ('DQN', 1, args.logdir+'DQN/'), ('Option', 2, args.logdir+'Option-2/'), ('Option', 4, args.logdir+'Option-4/'), ('Option', 8, args.logdir+'Option-8/')]

for type,num_option,folder in iteration:
    print("Model: ", type, '-', num_option)
    # if we want to iterate through all checkpoints
    # if type != 'random':
    #     if type == 'Option':
    #         files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and 'steps.zip' in f]
    #     else:
    #         files = [f.replace('.zip','') for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and 'steps.zip' in f]
    #     for f in files:
    #         print("Checkpoint: ", f.split('_')[-2])
    #         model = load_model(type, folder+f, num_option)
    #         get_stats(args, model, type, num_option)
    # # create the checkpoint
    if args.model is None:
        if type == "Option":
            checkpoint = "logs_gpu/grouped-standard-longrun/Option-4/" + "_5000000_steps.zip"
        else:
            # checkpoint = folder + "rl_model_1000000_steps"
            checkpoint = "logs_gpu/grouped-standard-longrun/DQN-cont-1mil-step/" + "rl_model_1200000_steps"
    else:
        checkpoint = folder + args.model
    # load model and other stuff
    # if model zip is none, we use a harded model zip
    if type == "Option" and num_option == 4:
        model = load_model(type, checkpoint, num_option)
        get_stats(args, model, type, num_option)
# game_over      = False
# obs            = env.reset()
# current_option = None
# # env.measure_step_time(verbose=True)
# env.render(wait_sec=1, mode='image', verbose=True)
# while not game_over:
#     # action, _ = model.predict(obs, deterministic=False)
#     # print("Action", action)
#     action, current_option = model.predict(obs, deterministic=False, current_option=current_option) # option critic version
#     print(f"Action: {action}| option: {current_option}")
#     obs, reward, game_over, info = env.step(action)
#
#     env.render(wait_sec=1, mode='image', verbose=True)
#     print("Reward", reward)
