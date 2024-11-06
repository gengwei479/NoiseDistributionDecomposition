import numpy as np
import torch


class Buffer:
    def __init__(self, arglist):
        self.N = arglist.N
        self.obs_dim = arglist.obs_dim
        self.state_dim = arglist.state_dim
        self.action_dim = arglist.action_dim
        self.episode_limit = arglist.episode_limit
        self.batch_size = arglist.PPO_v0_batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                       'var_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                       'avail_a_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'var': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'dw': np.ones([self.batch_size, self.episode_limit, self.N]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, obs_n, s, v_n, var_n, avail_a_n, a_n, a_logprob_n, r, var, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['var_n'][self.episode_num][episode_step] = var_n
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['var'][self.episode_num][episode_step] = var
        self.buffer['dw'][self.episode_num][episode_step] = np.array(dw).repeat(self.N)

        self.buffer['active'][self.episode_num][episode_step] = np.ones(self.N)

    def store_last_value(self, episode_step, v_n, var_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['var_n'][self.episode_num][episode_step] = var_n
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.long)
            elif key == 'v_n' or 'var_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len + 1], dtype=torch.float32)
            else:
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.float32)
        return batch