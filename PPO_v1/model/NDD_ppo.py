import numpy as np
import os
import time
import torch
import logging
from Common.log_process import Logger

from PPO_v1.model.normalization import RunningMeanStd, Normalization, RewardScaling
from PPO_v1.model.NDD_buffer import Buffer
from PPO_v1.model.NDD_agent import Agent

from DistDecompose.DD_networks import AgentDistribution, TotalDistributionDecompose
from DistDecompose.DiffusionGD import DiffusionGD

class NDD_Ppo_v1():
    def __init__(self, name, env, arglist, seed = 0):
        self.name = name
        self.arglist = arglist
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env = env
        self.env_info = self.env.get_env_info()
        self.arglist.N = self.env_info["n_agents"]
        self.arglist.obs_dim = self.env_info["obs_shape"]
        self.arglist.state_dim = self.env_info["state_shape"]
        self.arglist.action_dim = self.env_info["n_actions"]
        self.arglist.episode_limit = self.env_info["episode_limit"]

        self.agent_n = Agent(self.arglist)
        self.replay_buffer = Buffer(self.arglist)

        self.win_rates = []
        self.total_steps = 0
        if self.arglist.PPO_v0_use_reward_norm:
            self.reward_norm = Normalization(shape=1)
        elif self.arglist.PPO_v0_use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=self.arglist.PPO_v0_gamma)
        
        self.t_start = time.time()
        self.create_dir(arglist) 
        self.logger = Logger(str(self.name) + "_log_info_0000.log", str(os.getcwd()) + '/PPO_v1' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/')
        
        self.dist_decompose = TotalDistributionDecompose(self.arglist.N * arglist.DD_reward_partition_length, arglist.DD_reward_partition_length, 
                                                         [arglist.obs_dim for _ in range(self.arglist.N)], 
                                                          arglist.DD_agent_network_layer_dim, arglist.DD_network_learning_rate, arglist.reward_range, 
                                                          arglist.DD_reward_partition_length, arglist.DD_reward_sample_num, arglist.DD_episode_num, self.arglist.DD_epsilon)
        self.dGD = DiffusionGD(arglist)


    def main_run_smac(self):
        evaluate_num = -1
        rewards = []
        win_rates = []
        while self.total_steps < self.arglist.PPO_v0_max_train_steps:
            self.t_start = time.time()
            if self.total_steps // self.arglist.PPO_v0_evaluate_freq > evaluate_num:
                _rew, _win = self.evaluate_policy()
                rewards.append(_rew)
                win_rates.append(_win)
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.arglist.PPO_v0_batch_size:
                self.agent_n.learn(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        _rew, _win = self.evaluate_policy()
        rewards.append(_rew)
        win_rates.append(_win)
        self.env.close()
        
        return rewards, win_rates

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.arglist.PPO_v0_evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.arglist.PPO_v0_evaluate_times
        evaluate_reward = evaluate_reward / self.arglist.PPO_v0_evaluate_times
        self.win_rates.append(win_rate)
        
        print('episode', self.total_steps, 'score %.1f' % evaluate_reward, 'win rate %.1f' % win_rate, 'time_steps', round(time.time() - self.t_start, 3))
        self.logger.log_insert("env:{}, {}, steps: {}, avg score: {}, avg win rate: {}, time: {}".format(
            self.arglist.scene_name, self.name, self.total_steps, evaluate_reward, win_rate, round(time.time() - self.t_start, 3)), logging.INFO)
        
        self.save_model(self.arglist)
        return evaluate_reward, win_rate

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        if self.arglist.PPO_v0_use_reward_scaling:
            self.reward_scaling.reset()
        if self.arglist.PPO_v0_use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.arglist.episode_limit):
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)
            v_n, var_n = self.agent_n.get_value(s, obs_n)
            
            if self.arglist.DM_enable:
                r, done, info = self.env.step_sample(a_n, self.arglist.DM_input_dim)
                r[0] = self.dGD.dg(r[0], self.arglist.DD_reward_sample_num)
            else:
                r, done, info = self.env.step_sample(a_n, self.arglist.DD_reward_sample_num)
            
            #----------------------
            
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r[1]

            if not evaluate:
                if self.arglist.PPO_v0_use_reward_norm:
                    r = self.reward_norm(r)
                elif self.arglist.PPO_v0_use_reward_scaling:
                    r = self.reward_scaling(r)
                if done and episode_step + 1 != self.arglist.episode_limit:
                    dw = True
                else:
                    dw = False
                
                dists = self.dist_decompose.learn_dist(np.array(obs_n, dtype=np.float32), r[0])
                # rews = np.array(r[1]).repeat(self.arglist.N)
                rews = np.array([float(dists[i].mean) for i in range(self.arglist.N)])
                var = np.array([float(dists[i].variance) for i in range(self.arglist.N)])
                # print(str(r[1]) + ' --------------------------' + str(rews))
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, var_n, avail_a_n, a_n, a_logprob_n, rews, var, dw)

            if done:
                break

        if not evaluate:
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            v_n, var_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n, var_n)

        return win_tag, episode_reward, episode_step + 1

    def create_dir(self, arglist):
        if not os.path.exists(os.getcwd() + '/PPO_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/PPO_v1' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v1' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/PPO_v1' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v1' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
            
    def save_model(self, arglist):
        if not os.path.exists(os.getcwd() + '/PPO_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))

        torch.save({'actor_network': self.agent_n.actor.state_dict(), 'critic_network': self.agent_n.critic.state_dict(), 
                'var_network': self.agent_n.var.state_dict()}, 
                os.getcwd() + '/PPO_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/' + 
                str(arglist.scene_name) + '_' +
                str([arglist.noisy_configuration[i]['distribution']['type'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                str([arglist.noisy_configuration[i]['distribution']['param'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                str([arglist.noisy_configuration[i]['reward_influence'] for i in range(len(self.arglist.noisy_configuration))]) +
                '_agent_' + str(id) + '_model_param.pkl')
        pass