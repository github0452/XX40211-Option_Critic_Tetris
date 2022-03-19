import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from tetris import TetrisEnv
from option_critic import OptionCritic

# env = gym.make("Breakout-v0")
env = TetrisEnv(only_squares=True, grouped_actions=True, board_size=(10,20), no_rotations=True)
# model = DQN("CnnPolicy", env, verbose=0,buffer_size=10000, tensorboard_log='logs/DQN-tetris')
model = PPO("CnnPolicy", env, verbose=0, tensorboard_log='logs/PPO-tetris')
model = OptionCritic(env)
# pre-training
# game_over = False
# obs = env.reset()
# while not game_over:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, game_over, info = env.step(action)
#     env.render()
#     print("Game over:", game_over)

model = model.learn(total_timesteps=10000000, log_interval=10)
# model.save("ppo_tetris")
