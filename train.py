import argparse
from tetris.tetris import TetrisEnv
from stable_baselines3 import PPO, DQN
from option_critic.option_critic import OptionCritic, EvalCallbackOptionCritic, CheckpointCallbackOptionCritic
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def create_env(args, num_cpu=1):
    def make_env(board_size, args, seed=0):
        def _init():
            env = TetrisEnv(board_size=board_size, action_type=args.env_action_type, reward_type=args.env_reward_type, max_steps=args.env_max_steps, reward_scaling=args.env_reward_scaling)
            return env
        return _init
    env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])
    eval_env = DummyVecEnv([make_env(board_size, args) for i in range(num_cpu)])
    return env, eval_env

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
parser.add_argument('--env-size', default='20,10', help='size of the board')
parser.add_argument('--env-action-type', default='grouped', help='type of the action to take [grouped, semigrouped, standard]')
parser.add_argument('--env-reward-type', default='standard', help='what the reward type should be [standard, no piece drop, only lines, num lines]')
parser.add_argument('--env-reward-scaling', default=None, help='Should the reward be scaled [multi, log]')
parser.add_argument('--env-max-steps', default=18000, help='Maximum steps in environment (to prevent infinite runs)')
# learning arguments
parser.add_argument('--logdir', default='logs/test/', help='where should stuff be logged')
parser.add_argument('--device', default='cuda:0', help='What device should the model be trained on')
parser.add_argument('--save-freq', type=int, default=200000, help='how frequently should the model be checkpointed')
parser.add_argument('--eval-freq', type=int, default=2000, help='how frequently should the model be evaluated')
parser.add_argument('--logg-freq', type=int, default=100, help='how frequently should the model be logged')
parser.add_argument('--eval-num', type=int, default=10, help='how many episodes per model evaluation')
parser.add_argument('--training-num', type=int, default=1000000, help='how many episodes should the model be trained for')
# model arguments
parser.add_argument('--model-type', default="PPO", help='what model to be using [PPO, DQN, Option]')
parser.add_argument('--options', type=int, default=8, help='how many options')

args = parser.parse_args()
model_folder = args.logdir + args.model_type + "/"
board_size = tuple([int(i) for i in args.env_size.split(',')])

num_cpu = 1
env, eval_env = create_env(args, num_cpu)

if args.model_type == "PPO":
    model = create_PPO_Model(model_folder, args.device)
elif args.model_type == "DQN":
    model = create_DQN_Model(model_folder, args.device)
elif args.model_type == "Option":
    model = create_Option_Model(model_folder, args.device, args.options)
else:
    raise ValueError('Model type not recognized.')

save_freq    = max(args.save_freq // num_cpu, 1)
eval_freq    = max(args.eval_freq // num_cpu, 1)
logg_freq    = max(args.logg_freq // num_cpu, 1)
training_num = max(args.training_num // num_cpu, 1)
if args.model_type == "Option":
    checkpoint_callback = CheckpointCallbackOptionCritic(freq=save_freq, save_path=model_folder)
    eval_callback = EvalCallbackOptionCritic(eval_env, best_model_save_path=model_folder,
                                 log_path=model_folder, freq=eval_freq,
                                 deterministic=False, n_eval_episodes=args.eval_num)
else:
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_folder)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_folder,
                                 log_path=model_folder, eval_freq=eval_freq,
                                 deterministic=False, render=False, n_eval_episodes=args.eval_num)
model.learn(total_timesteps=training_num, log_interval=logg_freq, callback=[checkpoint_callback, eval_callback])
model.save(model_folder + "/final_model")
