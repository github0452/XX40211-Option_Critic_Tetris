import numpy as np
import random
from collections import deque

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

import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, logdir, run_name):
        number = 1
        self.log_name = logdir + '/' + run_name + '_'
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0

        while os.path.exists(self.log_name + str(number)): # keep incrementing until folder does not exist
            number += 1
        self.log_name += str(number)
        os.makedirs(self.log_name)
        self.writer = SummaryWriter(self.log_name)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
            )

    def log_train_episode(self, steps, reward, option_lengths, ep_steps, num_rand, epsilon):
        self.n_eps += 1
        # logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
        #     f"| hours={(time.time()-self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")
        self.writer.add_scalar(tag="rollout/episodic_rewards", scalar_value=reward, global_step=self.n_eps)
        self.writer.add_scalar(tag='rollout/episode_lengths', scalar_value=ep_steps, global_step=self.n_eps)
        self.writer.add_scalar(tag='rollout/percentage_random_action', scalar_value=num_rand/ep_steps, global_step=self.n_eps)
        # Keep track of options statistics
        for option, lens in option_lengths.items():
            # Need better statistics for this one, point average is terrible in this case
            self.writer.add_scalar(tag=f"rollout/options/option_{option}_avg_length", scalar_value=np.mean(lens) if len(lens)>0 else 0, global_step=self.n_eps)
            self.writer.add_scalar(tag=f"rollout/options/option_{option}_active", scalar_value=sum(lens)/ep_steps, global_step=self.n_eps)

    def log_eval_episode(self, avg_reward, avg_ep_steps, total_ep_steps, option_lengths):
        logging.info(f"> evaluation; ep {self.n_eps} done. avg_reward={avg_reward} | avg_episode_steps={avg_ep_steps} "\
            f"| hours={(time.time()-self.start_time) / 60 / 60:.3f}")
        self.writer.add_scalar(tag="eval/mean_ep_length", scalar_value=avg_ep_steps, global_step=self.n_eps)
        self.writer.add_scalar(tag='eval/mean_reward', scalar_value=avg_reward, global_step=self.n_eps)
        # Keep track of options statistics
        for option, lens in option_lengths.items():
            self.writer.add_scalar(tag=f"eval/options/option_{option}_avg_length", scalar_value=np.mean(lens) if len(lens)>0 else 0, global_step=self.n_eps)
            self.writer.add_scalar(tag=f"eval/options/option_{option}_active", scalar_value=sum(lens)/total_ep_steps, global_step=self.n_eps)

    def log_train_step(self, step, actor_loss, critic_loss, entropy, epsilon):
        if actor_loss:
            self.writer.add_scalar(tag="train/actor_loss", scalar_value=actor_loss.item(), global_step=step)
            loss = actor_loss if not critic_loss else actor_loss+critic_loss
            self.writer.add_scalar(tag="train/loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="train/critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="train/policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="train/epsilon",scalar_value=epsilon, global_step=step)
