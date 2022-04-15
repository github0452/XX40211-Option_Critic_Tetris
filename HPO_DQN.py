import optuna
from optuna.trial import TrialState
from tetris.tetris import TetrisEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

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

def create_model(trial):
    env = TetrisEnv(board_size=(10,10), action_type='grouped', output_type='image', simplified=False, score_reward=True, scale_reward=1.0, max_steps=18000)
    eval_env = TetrisEnv(board_size=(10,10), action_type='grouped', output_type='image', simplified=False, score_reward=True, scale_reward=1.0, max_steps=18000)
    # hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    tau = trial.suggest_uniform("tau", 0, 1)
    train_freq = trial.suggest_int("train_freq", 4, 100, log=True)
    target_update_interval = trial.suggest_int("target_update_interval", 200, 10000, log=True)
    exploration_fraction = trial.suggest_uniform("exploration_fraction ", 0.1, 1)
    buffer_size = trial.suggest_int("buffer_size ", 4096, 100000)
    f = open("dqn.txt", "a")
    f.write(f"trial num: {trial.number}, lr: {lr}, tau: {tau}, train_freq: {train_freq}, target_update_interval: {target_update_interval}, exploration_fraction: {exploration_fraction}, buffer_size: {buffer_size}")
    f.close()
    model = DQN("CnnPolicy", env, buffer_size=buffer_size, batch_size=512, learning_starts=4096,
        learning_rate=lr, tau=tau, gamma=0.99, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction,train_freq=train_freq,
        verbose=1, device='cuda:0', tensorboard_log="logs/HPO_DQN")
    return model, eval_env

def objective(trial):
    model, eval_env = create_model(trial)
    for step in range(5):
        if step == 0:
            model.learn(total_timesteps=120000, reset_num_timesteps=True) #40000
        else:
            model.learn(total_timesteps=120000, reset_num_timesteps=False) #40000
        avg_reward = evaluate_model(model, eval_env)
        trial.report(avg_reward, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    avg_reward = evaluate_model(model, eval_env)
    return avg_reward

study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner()
)
study.optimize(objective, n_trials=12)  # Invoke optimization of the objective function.
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))


for trial in study.trials:
    print("Trial ", trial.number)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
