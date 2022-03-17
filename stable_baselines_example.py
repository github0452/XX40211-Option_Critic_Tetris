import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from tetris import TetrisEnv

# env = gym.make("CartPole-v0")
env = TetrisEnv(only_squares=True, grouped_actions=True, board_size=(3,4))
# model = DQN("CnnPolicy", env, verbose=1,buffer_size=5000, tensorboard_log='logs/DQN')
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log='logs/PPO')
# pre-training
game_over = False
obs = env.reset()
while not game_over:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, game_over, info = env.step(action)
    env.render()
    print("Game over:", game_over)

for i in range(10):
    model = model.learn(total_timesteps=100000, n_eval_episodes=20, reset_num_timesteps=False, log_interval=1)
    model.save("dqn_cartpole")
    # model = DQN.load("dqn_cartpole", env)
    for _ in range(5):
        game_over = False
        obs = env.reset()
        while not game_over:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, game_over, info = env.step(action)
            env.render()
