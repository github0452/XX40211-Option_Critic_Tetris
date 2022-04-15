import optuna
from optuna.trial import TrialState
from tetris.tetris import TetrisEnv
from option_critic.option_critic import OptionCritic, EvalCallbackOptionCritic, CheckpointCallbackOptionCritic
from stable_baselines3.common.callbacks import CheckpointCallback

def evaluate_model(model, eval_env):
    rewards = 0
    for _ in range(100):
        game_over = False
        obs       = eval_env.reset()
        current_option = None
        while not game_over:
            action, current_option = model.predict(obs, deterministic=False, current_option=current_option) # option critic version
            obs, reward, game_over, _ = eval_env.step(action)
            rewards += reward
    avg_reward = rewards / 100
    return avg_reward

def create_model(trial):
    env = TetrisEnv(board_size=(10,10), action_type='grouped', output_type='image', simplified=False, score_reward=True, scale_reward=1.0, max_steps=18000)
    eval_env = TetrisEnv(board_size=(10,10), action_type='grouped', output_type='image', simplified=False, score_reward=True, scale_reward=1.0, max_steps=18000)
    # hyperparameters

    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    update_frequency = trial.suggest_int("update_frequency", 4, 100, log=True)
    freeze_interval = trial.suggest_int("freeze_interval", 200, 10000, log=True)
    temperature = trial.suggest_int("temperature", 1, 2000, log=True)
    entropy_reg = trial.suggest_float("entropy_reg", 0.001, 0.1, log=True)
    termination_reg = trial.suggest_float("termination_reg", 0.001, 0.1, log=True)
    batch_size=512
    buffer_size = trial.suggest_int("buffer_size ", 4096, 100000, log=True)
    epsilon_decay = trial.suggest_float("epsilon_decay", 2e3, 2e5, log=True)
    f = open("option_critic_HPO.txt", "a")
    f.write(f"trial num: {trial.number}, lr: {lr}, update_frequency: {update_frequency}, temperature: {temperature}, entropy_reg: {entropy_reg}, termination_reg: {termination_reg}, freeze_interval: {freeze_interval}, buffer_size: {buffer_size}, epsilon_decay: {epsilon_decay}")
    f.close()
    model = OptionCritic(env, num_options=8, learning_starts=4096,
        learning_rate=lr, update_frequency=update_frequency, temperature=temperature, entropy_reg=entropy_reg,
        termination_reg=termination_reg, freeze_interval=freeze_interval, batch_size=batch_size, buffer_size=buffer_size,
        epsilon_decay=epsilon_decay, gamma=0.99,
        logdir="logs/HPO_OptionCritic", device='cuda:3')
    return model, eval_env

def objective(trial):
    model, eval_env = create_model(trial)
    for step in range(5):
        model.learn(total_timesteps=60000)
        avg_reward = evaluate_model(model, eval_env)
        trial.report(avg_reward, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    avg_reward = evaluate_model(model, eval_env)
    return avg_reward

study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner()
)
study.optimize(objective, n_trials=6)  # Invoke optimization of the objective function.
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
