import time
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
        self.magic_number = 7744#7 * 7 * 64
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

        Q = self.get_options_q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.options_term_prob(state).sigmoid()

    def get_action(self, feature, option):
        logits = torch.matmul(feature, self.options_W[option]) + self.options_b[option]
        a_dist = (logits / self.temperature).softmax(dim=-1)
        a_dist = torch.distributions.Categorical(a_dist)
        a = a_dist.sample()
        logp = a_dist.log_prob(a)
        entropy = a_dist.entropy()
        return a.item(), logp, entropy

class OptionCritic():
    # parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
    # parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
    def __init__(self, env, num_options=2, temperature=1, seed=0, logdir='logs', entropy_reg=0.01, termination_reg=0.01,
            update_frequency=4, freeze_interval=200, batch_size=32, max_history=1000000,
            epsilon_decay=20000, epsilon_min=0.1, epsilon_start=1.0, gamma=0.99, learning_rate=0.00000025,
            exp="test"):
        self.env = env
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_options = num_options
        self.option_critic = OptionCriticConv(
            in_features=3,
            num_actions=self.env.action_space.n,
            num_options=num_options,
            temperature=temperature,
            device=self.device
        )
        self.option_critic_prime = deepcopy(self.option_critic)
        self.optim = torch.optim.RMSprop(self.option_critic.parameters(), lr=learning_rate)
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        self.buffer = ReplayBuffer(capacity=max_history, seed=seed)
        self.logger = Logger(logdir=logdir, run_name=f"option_critic-{type(env).__name__}-{exp}")
        self.gamma = gamma
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.freeze_interval = freeze_interval
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

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

    def learn(self, max_steps_total, max_steps_ep):
        steps = 0;
        while steps < max_steps_total:
            rewards = 0; option_lengths = {opt:[] for opt in range(self.num_options)}
            obs   = self.env.reset()
            state = self.option_critic.get_state_feature(torch.from_numpy(obs).float())
            greedy_option  = self.option_critic.get_greedy_options(state).item()
            game_over = False; ep_steps = 0; option_termination = True; curr_op_len = 0; current_option = 0
            while not game_over and ep_steps < max_steps_ep:
                epsilon = self.calc_epsilon(steps)

                if option_termination:
                    option_lengths[current_option].append(curr_op_len)
                    current_option = np.random.choice(self.num_options) if np.random.rand() < epsilon else greedy_option
                    curr_op_len = 0

                action, logp, entropy = self.option_critic.get_action(state, current_option)

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
                # update for next iteration
                state = self.option_critic.get_state_feature(torch.from_numpy(next_obs).float())
                option_termination, greedy_option = self.option_critic.predict_option_termination(state, current_option)
                # update for current episode tracking
                rewards += reward
                ep_steps += 1
                curr_op_len += 1
                obs = next_obs
                # update for global tracking
                steps += 1
                if steps % 4 == 0:
                    self.logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)
            self.logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)

if __name__=="__main__":
    env = TetrisEnv(board_size=(6,6), grouped_actions=True, only_squares=True, no_rotations=True, max_steps=500)
    model = OptionCritic(env)
    model.learn(max_steps_total=1000000, max_steps_ep=18000)
