import torch
import torch.nn as nn
from copy import deepcopy
from math import exp
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer(object):
    def __init__(self, capacity, seed):
        if seed is not None:
            self.rng = random.SystemRandom(seed)
        else:
            self.rng = random.SystemRandom(42)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
        return np.stack(obs), option, reward, np.stack(next_obs), done

    def __len__(self):
        return len(self.buffer)

class OptionCriticNet(nn.Module):
    def __init__(self, in_channels, num_options, num_actions, temperature, device):
        super(OptionCriticNet, self).__init__()
        self.magic_number = 21 * 46 * 64
        self.device = device
        self.cnn_feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )
        self.options_q_value = nn.Linear(512, num_options) # predicted q-value
        self.options_term_prob = nn.Linear(512, num_options) # probability it terminates
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
        self.temperature = temperature
        self.to(self.device)

    def get_state_feature(self, state):
        if state.ndim < 4:
            state = state.unsqueeze(0)
        return self.cnn_feature(state)

    def get_options_q(self, feature):
        return self.options_q_value(feature)

    def get_term_prob(self, feature, option):
        return self.options_term_prob(feature)[:, option]

    def sample_term_prob(self, feature, option):
        return torch.distributions.Bernoulli(self.get_term_prob(feature, option).sigmoid()).sample()

    def get_action(self, feature, option):
        logits = torch.matmul(feature, self.options_W[option]) + self.options_b[option]
        a_dist = (logits / self.temperature).softmax(dim=-1)
        a_dist = torch.distributions.Categorical(a_dist)
        a = a_dist.sample()
        logp = a_dist.log_prob(a)
        entropy = a_dist.entropy()
        return a.item(), logp, entropy

class OptionCritic():
    def __init__(self, env, num_options = 8, temperature = 1.0, termination_reg = 0.01, entropy_reg = 0.01, update_frequency = 4,
            learning_rate=0.0001, buffer_size=1000000,
            learning_starts=5000, batch_size=32, gamma=0.99, train_freq=4,
            target_update_interval=10000, eps_decay=int(1e6), exploration_initial_eps=1.0,
            exploration_final_eps=0.05, test_eps = 0, max_grad_norm=10, tensorboard_log='logs/OptionCritic',
            seed=None, device='auto'):
            # tau, gradient steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.tf_logger = SummaryWriter(tensorboard_log)
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            env.seed(self.seed)
        self.env = env
        #neural networks
        self.num_options = num_options
        self.option_critic = OptionCriticNet(
            in_channels = self.env.observation_space.shape[0],
            num_actions = self.env.action_space.n,
            num_options = self.num_options,
            temperature = temperature,
            device = self.device
        )
        self.option_critic_prime = deepcopy(self.option_critic) # Create a prime network for more stable Q values
        self.replay = ReplayBuffer(capacity=buffer_size, seed=self.seed)
        self.optim = torch.optim.RMSprop(self.option_critic.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg

        self.gamma = gamma
        self.train_freq = train_freq
        self.max_grad_norm = max_grad_norm
        self.learning_starts = learning_starts
        self.update_frequency = update_frequency
        self.target_update_interval = target_update_interval

        self.eps_start = exploration_initial_eps
        self.eps_final = exploration_final_eps
        self.eps_decay = eps_decay
        self.test_eps = test_eps

    def calc_epsilon(self, steps):
        return self.eps_final + (self.eps_start - self.eps_final) * exp(-steps / self.eps_decay)

    def conv_obs_to_tensor(self, obs):
        return torch.from_numpy(obs).float().to(self.device)

    def calc_critic_loss(self, data_batch):
        obs, options, rewards, next_obs, dones = data_batch
        options   = torch.LongTensor(options).to(self.device)
        rewards   = torch.FloatTensor(rewards).to(self.device)
        masks     = 1 - torch.FloatTensor(dones).to(self.device)
        obs       = self.conv_obs_to_tensor(obs)
        next_obs  = self.conv_obs_to_tensor(next_obs)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        features  = self.option_critic.get_state_feature(obs).squeeze(0)
        q  = self.option_critic.get_options_q(features)

        # the update target contains Q_next, but for stable learning we use prime network for this
        next_features_prime = self.option_critic_prime.get_state_feature(next_obs).squeeze(0)
        next_q_prime      = self.option_critic_prime.get_options_q(next_features_prime)
        next_option_q_prime = next_q_prime[:,options]
        next_best_q_prime = next_q_prime.max(dim=-1)[0]

        # Additionally, we need the beta probabilities of the next state
        next_features          = self.option_critic.get_state_feature(next_obs).squeeze(0)
        next_options_term_prob = self.option_critic.get_term_prob(next_features, options)

        # Now we can calculate the update target gt
        update_target = rewards + masks * self.gamma * \
            ((1 - next_options_term_prob) * next_option_q_prime + next_options_term_prob  * next_best_q_prime)

        # to update Q we want to use the actual network, not the prime
        td_err = (q[:, options] - update_target.detach()).pow(2).mul(0.5).mean()
        return td_err

    def calc_actor_loss(self, obs, option, reward, next_obs, done, logp, entropy):
        obs = self.conv_obs_to_tensor(obs)
        next_obs = self.conv_obs_to_tensor(next_obs)

        next_feature = self.option_critic.get_state_feature(next_obs)
        next_term_prob = self.option_critic.get_term_prob(next_feature, option)

        next_feature_prime = self.option_critic_prime.get_state_feature(next_obs)
        next_q_prime = self.option_critic_prime.get_options_q(next_feature_prime).squeeze()
        next_option_q_prime = next_q_prime[option]
        next_best_q_prime = next_q_prime.max(dim=-1)[0]

        update_target = reward + (1 - done) * self.gamma * \
            ((1 - next_term_prob) * next_option_q_prime + next_term_prob  * next_best_q_prime)

        feature = self.option_critic.get_state_feature(obs)
        q = self.option_critic.get_options_q(feature).squeeze().detach()
        option_q = q[option];      best_q = q.max(dim=-1)[0]
        term_prob = self.option_critic.get_term_prob(feature, option)

        policy_loss = -logp * (update_target.detach() - option_q) - self.entropy_reg * entropy
        termination_loss = term_prob * (option_q.detach() - best_q.detach() + self.termination_reg) * (1 - done)
        actor_loss = termination_loss + policy_loss
        return actor_loss

    def predict(observation):
        pass

    def evaluate():
        pass

    def learn(self, total_timesteps, log_interval=4, eval_freq=- 1, n_eval_episodes=5, tb_log_name='DQN', eval_log_path=None):
        steps = 0
        ep_count = 0
        # loggin stuff
        episode_rewards   = 0
        episode_loss = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        episode_entropy = 0
        episode_steps = 0
        while steps < total_timesteps: # training loop
            # env stuff
            game_over = False
            obs       = self.env.reset()
            feature   = self.option_critic.get_state_feature(self.conv_obs_to_tensor(obs)) # pass through conv network
            # option stuff
            current_option = 0
            option_termination = True
            greedy_option = self.option_critic.get_options_q(feature).argmax(dim=-1)


            while not game_over:
                epsilon = self.calc_epsilon(steps)
                # option selection
                if option_termination:
                    current_option = np.random.choice(self.num_options) if np.random.rand() < epsilon else greedy_option
                # action selection
                action, logp, entropy = self.option_critic.get_action(feature, current_option)
                next_obs, reward, game_over, _ = self.env.step(action)
                # add to replay buffer
                self.replay.push(obs, current_option, reward, next_obs, game_over)
                # update before backpropagation?
                feature = self.option_critic.get_state_feature(self.conv_obs_to_tensor(obs)) # extract feature vector again
                option_termination = self.option_critic.sample_term_prob(feature, current_option) # determine whether or not it ends
                greedy_option = self.option_critic.get_options_q(feature).argmax(dim=-1)
                # backpropagation and loss calc
                actor_loss, critic_loss, loss = None, None, None
                if steps > self.learning_starts and steps % self.train_freq == 0:
                    actor_loss = self.calc_actor_loss(obs, current_option, reward, next_obs, game_over, logp, entropy)
                    loss = actor_loss
                    if steps % self.update_frequency == 0:
                        databatch = self.replay.sample(self.batch_size)
                        critic_loss = self.calc_critic_loss(databatch) # calculate critic loss
                        loss += critic_loss
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.option_critic.parameters(), self.max_grad_norm)
                    self.optim.step()
                    if steps % self.target_update_interval == 0:
                        self.option_critic_prime.load_state_dict(self.option_critic.state_dict())
                # update global steps etc
                steps += 1
                obs = next_obs
                # logging stuff
                episode_rewards += reward
                episode_entropy += entropy.item()
                if loss is not None:
                    episode_loss += loss.item()
                if actor_loss is not None:
                    episode_actor_loss += actor_loss.item()
                if critic_loss is not None:
                    episode_critic_loss += critic_loss.item()
                episode_steps += 1
            if ep_count % log_interval == 0:
                if episode_loss != 0:
                    self.tf_logger.add_scalar('train/loss', episode_loss/log_interval, ep_count)
                if episode_actor_loss != 0:
                    self.tf_logger.add_scalar('train/actor_loss', episode_actor_loss/log_interval, ep_count)
                if episode_critic_loss != 0:
                    self.tf_logger.add_scalar('train/critic_loss', episode_critic_loss/log_interval, ep_count)
                self.tf_logger.add_scalar('train/entropy', episode_entropy/log_interval, ep_count)
                self.tf_logger.add_scalar('train/episode_steps', episode_entropy/log_interval, ep_count)
                self.tf_logger.add_scalar('train/rewards', episode_rewards/log_interval, ep_count)
                episode_rewards   = 0
                episode_loss = 0
                episode_actor_loss = 0
                episode_critic_loss = 0
                episode_entropy = 0
                episode_steps = 0
            ep_count += 1
