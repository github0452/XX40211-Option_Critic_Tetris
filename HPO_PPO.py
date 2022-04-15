import optuna
from optuna.trial import TrialState
from tetris.tetris import TetrisEnv
from stable_baselines3 import PPO
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
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.75, 0.95)
    max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.1, 0.9)
    ent_coef = trial.suggest_uniform("ent_coef", 0, 1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    n_steps= trial.suggest_int("n_steps", 10, 1000, log=True)
    f = open("ppo.txt", "a")
    f.write(f"trial num: {trial.number}, lr: {lr}, gae_lambda: {gae_lambda}, max_grad_norm: {max_grad_norm}, ent_coef: {ent_coef}, vf_coef: {vf_coef}, n_steps: {n_steps}")
    f.close()
    model = PPO("CnnPolicy", env, batch_size=512,
        gamma=0.99, max_grad_norm=max_grad_norm, gae_lambda=gae_lambda,
        learning_rate=lr, n_steps=n_steps,
        ent_coef=ent_coef, vf_coef=vf_coef,
        verbose=1, tensorboard_log="logs/HPO_PPO", device='cuda:1')
    return model, eval_env

def objective(trial):
    model, eval_env = create_model(trial)
    for step in range(5):
        if step == 0:
            model.learn(total_timesteps=60000, reset_num_timesteps=True) #40000
        else:
            model.learn(total_timesteps=60000, reset_num_timesteps=False) #40000
        avg_reward = evaluate_model(model, eval_env)
        trial.report(avg_reward, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    avg_reward = evaluate_model(model, eval_env)
    return avg_reward

study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner()
)
study.optimize(objective, n_trials=32)  # Invoke optimization of the objective function.
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
