import time
import os
import numpy as np
import argparse
from copy import deepcopy
from math import exp

from util import ReplayBuffer, Logger
from tetris import TetrisEnv

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

class OptionCriticConv(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                device='cpu',
                testing=False):

        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 26496#7 * 7 * 64
        self.device = device

        self.cnn_feature = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )
        self.options_q_value            = nn.Linear(512, num_options)                 # Policy-Over-Options
        self.options_term_prob = nn.Linear(512, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
        self.temperature = temperature
        self.to(self.device)

    def get_state_feature(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.cnn_feature(obs)
        return state

    def get_options_q(self, state):
        return self.options_q_value(state)

    def get_greedy_options(self, feature):
        return self.get_options_q(feature).argmax(dim=-1)

    def predict_option_termination(self, state, current_option):
        termination = self.options_term_prob(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        return bool(option_termination.item())

    def get_terminations(self, state):
        return self.options_term_prob(state).sigmoid()

    def get_action(self, feature, option, training=False):
        logits = torch.matmul(feature, self.options_W[option]) + self.options_b[option]
        a_dist = (logits / self.temperature).softmax(dim=-1)
        a_dist = torch.distributions.Categorical(a_dist)
        a = a_dist.sample()
        if not training:
            return a.item()
        else:
            logp = a_dist.log_prob(a)
            entropy = a_dist.entropy()
            return a.item(), logp, entropy

class EvalCallbackOptionCritic():
    def __init__(self, eval_env, best_model_save_path, log_path, freq, deterministic=False, n_eval_episodes=100, eval_epsilon=0.05, max_steps_ep=18000):
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.n_eval_episodes = n_eval_episodes
        self.max_steps_ep = max_steps_ep
        self.eval_epsilon = eval_epsilon
        self.best_reward = -np.inf
        self.freq = freq
        if not os.path.exists(self.best_model_save_path):
            os.makedirs(self.best_model_save_path)

class CheckpointCallbackOptionCritic():
    def __init__(self, freq, save_path, name_prefix=''):
        self.freq = freq
        self.checkpoint_prefix = save_path + name_prefix
        if not os.path.exists(save_path):
            os.makedirs(save_path)

class OptionCritic():
    # parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
    # parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
    def __init__(self, env, num_options=2, temperature=1, seed=0, logdir='logs', entropy_reg=0.01, termination_reg=0.01,
            update_frequency=4, freeze_interval=200, batch_size=32, buffer_size=1000000,
            epsilon_decay=20000, epsilon_min=0.1, epsilon_start=1.0, gamma=0.99, learning_rate=0.00000025, device='cuda:0'):
        self.env = env
        self.device = torch.device(device)
        self.seed = self.set_seed(seed)
        self.option_critic = OptionCriticConv(
            in_features=3,
            num_actions=self.env.action_space.n,
            num_options=num_options,
            temperature=temperature,
            device=self.device
        )
        self.option_critic_prime = deepcopy(self.option_critic)
        self.optim = torch.optim.RMSprop(self.option_critic.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        self.logger = Logger(logdir=logdir, run_name=type(self).__name__)
        self.gamma = gamma
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.freeze_interval = freeze_interval
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)
        return seed

    def calc_epsilon(self, steps):
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) * exp(-steps / self.epsilon_decay)

    def calc_critic_loss(self, data_batch):
        obs, options, rewards, next_obs, game_overs = data_batch
        batch_idx = torch.arange(len(options)).long()
        options   = torch.LongTensor(options).to(self.device)
        rewards   = torch.FloatTensor(rewards).to(self.device)
        masks     = 1 - torch.FloatTensor(game_overs).to(self.device)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        states = self.option_critic.get_state_feature(torch.from_numpy(obs).float()).squeeze(0)
        Q      = self.option_critic.get_options_q(states)

        # the update target contains Q_next, but for stable learning we use prime network for this
        next_states_prime = self.option_critic_prime.get_state_feature(torch.from_numpy(next_obs).float()).squeeze(0)
        next_Q_prime      = self.option_critic_prime.get_options_q(next_states_prime) # detach?

        # Additionally, we need the beta probabilities of the next state
        next_states            = self.option_critic.get_state_feature(torch.from_numpy(next_obs).float()).squeeze(0)
        next_termination_probs = self.option_critic.get_terminations(next_states).detach()
        next_options_term_prob = next_termination_probs[batch_idx, options]

        # Now we can calculate the update target gt
        gt = rewards + masks * self.gamma * \
            ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

        # to update Q we want to use the actual network, not the prime
        td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
        return td_err

    def calc_actor_loss(self, obs, option, logp, entropy, reward, game_over, next_obs):
        state = self.option_critic.get_state_feature(torch.from_numpy(obs).float())
        next_state = self.option_critic.get_state_feature(torch.from_numpy(next_obs).float())
        next_state_prime = self.option_critic_prime.get_state_feature(torch.from_numpy(next_obs).float())

        option_term_prob = self.option_critic.get_terminations(state)[:, option]
        next_option_term_prob = self.option_critic.get_terminations(next_state)[:, option].detach()

        Q = self.option_critic.get_options_q(state).detach().squeeze()
        next_Q_prime = self.option_critic_prime.get_options_q(next_state_prime).detach().squeeze()

        # Target update gt
        gt = reward + (1 - game_over) * self.gamma * \
            ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

        # The termination loss
        termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + self.termination_reg) * (1 - game_over)

        # actor-critic policy gradient with entropy regularization
        policy_loss = -logp * (gt.detach() - Q[option]) - self.entropy_reg * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss

    def predict(self, obs, current_option=None, deterministic=False, epsilon=0.05):
        # get state
        state = self.option_critic.get_state_feature(torch.from_numpy(obs).float())
        # determine whether or not option is terminating
        if current_option is None:
            option_termination = True
        else:
            option_termination = self.option_critic.predict_option_termination(state, current_option)
        # if terminating, we want to determine current option, otherwise it remains the same
        if option_termination:
            if not deterministic and np.random.rand() < epsilon:
                current_option = np.random.choice(self.option_critic.num_options)
            else:
                current_option = self.option_critic.get_greedy_options(state).item()
        # select action
        action = self.option_critic.get_action(state, current_option, training=False)
        return action, current_option

    def evaluate(self, evalCallback):
        # self.best_model_save_path = best_model_save_path
        rewards        = 0
        option_lengths = {opt:[] for opt in range(self.option_critic.num_options)}
        total_ep_steps = 0
        for _ in range(evalCallback.n_eval_episodes):
            obs            = evalCallback.eval_env.reset()
            game_over = False; ep_steps = 0; option_termination = True; current_option = None
            while not game_over and ep_steps < evalCallback.max_steps_ep:
                state = self.option_critic.get_state_feature(torch.from_numpy(obs).float())
                # determine whether or not option is terminating
                if current_option is None:
                    option_termination = True
                else:
                    option_termination = self.option_critic.predict_option_termination(state, current_option)
                # if terminating, we want to determine current option, otherwise it remains the same
                if option_termination:
                    if current_option is not None:
                        option_lengths[current_option].append(curr_op_len)
                    curr_op_len = 0
                    if not evalCallback.deterministic and np.random.rand() < evalCallback.eval_epsilon:
                        current_option = np.random.choice(self.option_critic.num_options)
                    else:
                        current_option = self.option_critic.get_greedy_options(state).item()
                # select action
                action = self.option_critic.get_action(state, current_option, training=False)
                next_obs, reward, game_over, _ = evalCallback.eval_env.step(action)
                # update for next iteration
                rewards += reward
                ep_steps += 1
                curr_op_len += 1
                obs = next_obs
            total_ep_steps += ep_steps
        avg_reward = rewards / evalCallback.n_eval_episodes
        avg_ep_steps = total_ep_steps / evalCallback.n_eval_episodes
        self.logger.log_eval_episode(avg_reward, avg_ep_steps, total_ep_steps, option_lengths)
        if avg_reward > evalCallback.best_reward:
            evalCallback.best_reward = avg_reward
            self.save(evalCallback.best_model_save_path + "best_model.zip")

    def save(self, path):
        torch.save({
            'model_option_critic': self.option_critic.state_dict(),
            'optim': self.optim.state_dict(),
            'seed': self.seed
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.option_critic.load_state_dict(checkpoint['model_option_critic'])
        self.seed = self.set_seed(checkpoint['seed'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.option_critic_prime = deepcopy(self.option_critic)

    def checkpoint(self, checkpointCallback, steps):
        self.save(f"{checkpointCallback.checkpoint_prefix}_{steps}_steps.zip")

    def learn(self, total_timesteps, log_interval=1, callback=[]):
        steps = 0;
        while steps < max_steps_total:
            rewards        = 0
            option_lengths = {opt:[] for opt in range(self.option_critic.num_options)}
            obs            = self.env.reset()
            game_over = False; ep_steps = 0; option_termination = True; current_option = None; num_rand = 0
            while not game_over and ep_steps < max_steps_ep:
                # get state
                state = self.option_critic.get_state_feature(torch.from_numpy(obs).float())
                # determine whether or not option is terminating
                if current_option is None:
                    option_termination = True
                else:
                    option_termination = self.option_critic.predict_option_termination(state, current_option)
                # if terminating, we want to determine current option, otherwise it remains the same
                epsilon = self.calc_epsilon(steps)
                if option_termination:
                    if current_option is not None:
                        option_lengths[current_option].append(curr_op_len)
                    curr_op_len = 0
                    if np.random.rand() < epsilon:
                        current_option = np.random.choice(self.option_critic.num_options)
                        num_rand += 1
                    else:
                        current_option = self.option_critic.get_greedy_options(state).item()
                # select action
                action, logp, entropy = self.option_critic.get_action(state, current_option, training=True)
                next_obs, reward, game_over, _ = self.env.step(action)
                self.buffer.push(obs, current_option, reward, next_obs, game_over)

                actor_loss, critic_loss = None, None
                if len(self.buffer) > self.batch_size:
                    actor_loss = self.calc_actor_loss(obs, current_option, logp, entropy, \
                        reward, game_over, next_obs)
                    loss = actor_loss
                    if steps % self.update_frequency == 0:
                        data_batch = self.buffer.sample(self.batch_size)
                        critic_loss = self.calc_critic_loss(data_batch)
                        loss += critic_loss
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    if steps % self.freeze_interval == 0:
                        self.option_critic_prime.load_state_dict(self.option_critic.state_dict())
                # update for current episode tracking
                rewards += reward
                ep_steps += 1
                curr_op_len += 1
                obs = next_obs
                # update for global tracking
                steps += 1
                if steps % log_interval == 0:
                    self.logger.log_train_step(steps, actor_loss, critic_loss, entropy.item(), epsilon)
                for cb in callback:
                    if steps % cb.freq == 0:
                        if isinstance(cb, EvalCallbackOptionCritic):
                            self.evaluate(cb)
                        elif isinstance(cb, CheckpointCallbackOptionCritic):
                            self.checkpoint(cb, steps)
            self.logger.log_train_episode(steps, rewards, option_lengths, ep_steps, num_rand, epsilon)

if __name__=="__main__":
    env = TetrisEnv(board_size=(6,6), grouped_actions=True, only_squares=True, no_rotations=True, max_steps=500)
    model = OptionCritic(env)
    model.learn(max_steps_total=1000000, max_steps_ep=18000)
