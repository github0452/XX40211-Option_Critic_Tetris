def evaluate_model(model, eval_env):
    rewards = 0
    for _ in range(100):
        game_over = False
        obs       = env.reset()
        while not game_over:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, game_over, _ = env.step(action)
            rewards += reward
    avg_reward = rewards / 100
    return avg_reward

def create_model(trial, "DQN"):
    env = TetrisEnv(board_size=(10,10), action_type='grouped_actions', output_type='image', simplified=False, score_reward=True, scale_reward=1.0, max_steps=18000)
    eval_env = TetrisEnv(board_size=(10,10), action_type='grouped_actions', output_type='image', simplified=False, score_reward=True, scale_reward=1.0, max_steps=18000)
    # hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    tau = trial.suggest_float("tau", 0, 1, log=True)
    gamma = trial.suggest("gamma", 0.5, 1, log=True)
    target_update_interval = ("target_update_interval", 1, 100, log=True)
    exploration_fraction = ("exploration_fraction ", 0.1, 1, log=True)
    model = DQN("CnnPolicy", env, buffer_size=100000, batch_size=512, learning_starts=2048
        learning_rate=lr, tau=tau, gamma=gamma, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction,
        verbose=1, device='cuda:0')
    return model, eval_env

def objective(trial):
    model, eval_env = create_model(trial)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=model_folder)
    for step in range(5):
        model.learn(total_timesteps=100000, callback=[checkpoint_callback])
        avg_reward = evaluate_model(model, eval_env)
        trial.report(avg_reward, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    avg_reward = evaluate_model(model, eval_env)
    return avg_reward

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=2)  # Invoke optimization of the objective function.
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
