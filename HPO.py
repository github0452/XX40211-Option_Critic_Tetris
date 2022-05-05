import optuna
from optuna.trial import TrialState
from tetris.tetris import TetrisEnv
from stable_baselines3 import PPO, DQN
from option_critic.option_critic import OptionCritic
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def create_env(num_cpu=1):
    def make_env(board_size, seed=0):
        def _init():
            env = TetrisEnv(board_size=board_size, action_type='grouped', reward_type='standard', max_steps=18000, reward_scaling=None)
            return env
        return _init
    env = DummyVecEnv([make_env((10,10)) for i in range(num_cpu)])
    eval_env = make_env((10,10))()
    return env, eval_env

def evaluate_model(model, eval_env):
    rewards = 0
    for _ in range(100):
        game_over = False
        obs       = eval_env.reset()
        while not game_over:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, game_over, _ = eval_env.step(action)
            rewards += reward
    avg_reward = rewards / 100
    return avg_reward

def create_Option_model(trial):
    env, eval_env = create_env()
    # hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    update_frequency = trial.suggest_int("update_frequency", 4, 100, log=True)
    freeze_interval = trial.suggest_int("freeze_interval", 200, 10000, log=True)
    temperature = trial.suggest_int("temperature", 1, 2000, log=True)
    entropy_reg = trial.suggest_float("entropy_reg", 0.01, 0.1, log=True)
    termination_reg = trial.suggest_float("termination_reg", 0.01, 0.1, log=True)
    batch_size=512
    buffer_size = 50000
    epsilon_decay = trial.suggest_float("epsilon_decay", 2e3, 2e7, log=True)
    model = OptionCritic(env, num_options=8, learning_starts=4096,
        lr=lr, update_frequency=update_frequency, temperature=temperature, entropy_reg=entropy_reg,
        termination_reg=termination_reg, freeze_interval=freeze_interval, batch_size=batch_size, buffer_size=buffer_size,
        epsilon_decay=epsilon_decay, gamma=0.99,
        logdir="logs/HPO_OptionCritic", device=args.device)
    return model, eval_env

def create_DQN_model(trial):
    env, eval_env = create_env()
    # hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    tau = trial.suggest_uniform("tau", 0, 1)
    train_freq = trial.suggest_int("train_freq", 4, 100, log=True)
    target_update_interval = trial.suggest_int("target_update_interval", 200, 10000, log=True)
    exploration_fraction = trial.suggest_uniform("exploration_fraction ", 0.1, 1)
    buffer_size = 50000
    model = DQN("CnnPolicy", env, buffer_size=buffer_size, batch_size=512, learning_starts=4096,
        learning_rate=lr, tau=tau, gamma=0.99, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction,train_freq=train_freq,
        verbose=1, device=args.device, tensorboard_log="logs/HPO_DQN")
    return model, eval_env

def create_PPO_model(trial):
    env, eval_env = create_env()
    # hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.75, 0.95)
    max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.1, 0.9)
    ent_coef = trial.suggest_uniform("ent_coef", 0, 1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    n_steps= trial.suggest_int("n_steps", 10, 1000, log=True)
    model = PPO("CnnPolicy", env, batch_size=512,
        gamma=0.99, max_grad_norm=max_grad_norm, gae_lambda=gae_lambda,
        learning_rate=lr, n_steps=n_steps,
        ent_coef=ent_coef, vf_coef=vf_coef,
        verbose=1, tensorboard_log="logs/HPO_PPO", device=args.device)
    return model, eval_env

def create_objective(model_type):
    create_model_fn = {
        'PPO': create_PPO_model,
        'DQN': create_DQN_model,
        'Option': create_Option_model
    }
    create_model = create_model_fn[model_type]
    def objective(trial):
        model, eval_env = create_model(trial)
        for step in [10000, 15000, 25000, 500000]:
            if step == 10000: # if first step
                model.learn(total_timesteps=step, reset_num_timesteps=True)
            else:
                model.learn(total_timesteps=step, reset_num_timesteps=False) #40000
            avg_reward = evaluate_model(model, eval_env)
            trial.report(avg_reward, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        avg_reward = evaluate_model(model, eval_env)
        f = open(model_type+"_HPO.txt", "a")
        f.write(f"trial num: {trial.number}, {str(trial.params)}")
        f.close()
        return avg_reward
    return objective

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# environment arguments#
parser.add_argument('--model-type', default="PPO", help='what model to be using [PPO, DQN, Option]')
parser.add_argument('--device', default='cuda:0', help='What device should the model be trained on')

args = parser.parse_args()

study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner()
)
study.optimize(create_objective(args.model_type), n_trials=32)  # Invoke optimization of the objective function.
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
