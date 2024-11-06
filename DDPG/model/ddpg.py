import numpy as np
import os
import time
import torch
import logging
from copy import deepcopy

from Common.log_process import Logger
from DDPG.model.maddpg import MADDPG
from DDPG.model.matd3 import MATD3
from DDPG.model.replay_buffer import ReplayBuffer

class Ddpg_v1():
    def __init__(self, name, env, arglist, seed = 0):
        self.name = name
        self.arglist = arglist
        self.cur_algo = str.upper(arglist.algorithm)
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env = env
        self.env_info = self.env.get_env_info()
        self.arglist.N = self.env_info["n_agents"]
        # self.arglist.obs_dim = self.env_info["obs_shape"]
        # self.arglist.state_dim = self.env_info["state_shape"]
        # self.arglist.action_dim = self.env_info["n_actions"]
        self.arglist.episode_limit = self.env_info["episode_limit"]
        
        self.agents = []
        self.win_rates = []
        self.total_steps = 0
        self.replay_buffer = ReplayBuffer(self.arglist)
        self.create_agents(self.arglist)
        
        self.create_dir(arglist)
        self.logger = Logger(str(self.name) + "_log_info_0000.log", str(os.getcwd()) + '/' + self.cur_algo + '_v2' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/')
        pass
    
    def main_run_smac(self):
        evaluate_num = -1
        rewards = []
        win_rates = []
        while self.total_steps < self.arglist.DDPG_v1_max_train_steps:
            self.t_start = time.time()
            if self.total_steps // self.arglist.DDPG_v1_evaluate_freq > evaluate_num:
                _rew, _win = self.evaluate_policy()
                rewards.append(_rew)
                win_rates.append(_win)
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.arglist.DDPG_v2_buffer_size:
                for id, agent in enumerate(self.agents):
                    agent.train(self.replay_buffer, self.agents)
                # self.agent_n.learn(self.replay_buffer, self.total_steps)
                # self.replay_buffer.reset_buffer()
            

        _rew, _win = self.evaluate_policy()
        rewards.append(_rew)
        win_rates.append(_win)
        self.env.close()
        
        return rewards, win_rates

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.arglist.DDPG_v1_evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.arglist.DDPG_v1_evaluate_times
        evaluate_reward = evaluate_reward / self.arglist.DDPG_v1_evaluate_times
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
        for episode_step in range(self.arglist.episode_limit):
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            a_n = []
            for id, agent in enumerate(self.agents):
                _a = agent.choose_action(obs_n[id], noise_std=0)
                a_n.append(np.argmax(np.array(_a) * np.array(avail_a_n[id])))
            # a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)
            
            # v_n = self.agent_n.get_value(s, obs_n)
            r, done, info = self.env.step(a_n)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if done and episode_step + 1 != self.arglist.episode_limit:
                    dw = True
                else:
                    dw = False
                    
                rews = np.array(r).repeat(self.arglist.N)
                obs_prime_n = self.env.get_obs()
                
                # if len(states) != 0 and len(actions) != 0 and len(rewards) != 0 and len(states_prime) != 0:
                self.replay_buffer.store_transition(obs_n, a_n, rews, obs_prime_n, [dw] * self.arglist.N)
                # self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, rews, dw)

            if done:
                break

        # if not evaluate:
        #     obs_n = self.env.get_obs()
        #     s = self.env.get_state()
        #     v_n = self.agent_n.get_value(s, obs_n)
        #     self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return win_tag, episode_reward, episode_step + 1
    
    def create_agents(self, arglist):
        if arglist.env == "smac":
            for i in range(arglist.N):
                if arglist.algorithm == 'ddpg':
                    self.agents.append(MADDPG(arglist, i))
                elif arglist.algorithm == 'td3':
                    self.agents.append(MATD3(arglist, i))
    
    def create_dir(self, arglist):
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
           
    def save_model(self, arglist):
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        for id, agent in enumerate(self.agents):
            torch.save({'actor_network': agent.actor.state_dict(), 
                    'critic_network': agent.critic.state_dict()}, 
                    os.getcwd() + '/' + self.cur_algo + '_v1' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/' + 
                    str(arglist.scene_name) + '_' +
                    str([arglist.noisy_configuration[i]['distribution']['type'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                    str([arglist.noisy_configuration[i]['distribution']['param'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                    str([arglist.noisy_configuration[i]['reward_influence'] for i in range(len(self.arglist.noisy_configuration))]) +
                    '_agent_' + str(id) + '_model_param.pkl')
        pass


class Ddpg_v2():
    def __init__(self, name,  env, arglist, seed = 0) -> None:
        self.name = name
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.cur_algo = str.upper(arglist.algorithm)
        self.agents = []
        self.arglist = arglist
        self.arglist.N = sum(arglist.agent_num)
        self.env = env
        self.env.reset()
        
        self.max_episodes_num = arglist.DDPG_v2_max_episodes_num
        self.replay_buffer = ReplayBuffer(self.arglist)
        self.noise_std = arglist.DDPG_v2_noise_std_init
        
        self.create_agents(self.arglist)
        self.create_dir(arglist)
        self.logger = Logger(str(self.name) + "_log_info_0000.log", str(os.getcwd()) + '/' + self.cur_algo + '_v2' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/')
        
    def main_run_mpe(self):
        score_history = []
        result_scores = []
        
        learn_iters = 0
        avg_score = 0
        n_steps = 0
        
        for i in range(self.max_episodes_num):
            score = 0
            t_start = time.time()
            states = self.env.reset()
            states_prime = {}
            while self.env.agents:
                actions = {}
                for id, agent in enumerate(self.agents):
                    actions[self.env.agents[id]] = agent.choose_action(states[self.env.agents[id]], noise_std=0)
                states_prime, rewards, terminations, truncations, infos = self.env.step(actions)
                
                if len(self.env.agents) != 0:
                    reward = sum(list(rewards.values()))
                
                n_steps += 1
                score += sum(list(rewards.values()))

                if len(states) != 0 and len(actions) != 0 and len(rewards) != 0 and len(states_prime) != 0:
                    self.replay_buffer.store_transition(list(states.values()), list(actions.values()), list(rewards.values()), list(states_prime.values()), list(truncations.values()))
                if self.replay_buffer.current_size >= self.arglist.DDPG_v2_batch_size:
                    for id, agent in enumerate(self.agents):
                        agent.train(self.replay_buffer, self.agents)
                        learn_iters += 1
                        
                if self.arglist.DDPG_v2_use_noise_decay:
                    noise_std_decay = (self.arglist.DDPG_v2_noise_std_init - self.arglist.DDPG_v2_noise_std_min) / self.arglist.DDPG_v2_noise_decay_steps
                    self.noise_std = self.noise_std - noise_std_decay if self.noise_std - noise_std_decay > self.arglist.DDPG_v2_noise_std_min else self.arglist.DDPG_v2_noise_std_min
                
                states = deepcopy(states_prime)
                states_prime = {}
            score_history.append(score)
            print(str(len(score_history)) + "--------------------")
            if len(score_history) >= 100:
                avg_score = np.mean(score_history[-100:])
                result_scores.append(avg_score)
                print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
                self.logger.log_insert("env:{}, {}, steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(self.arglist.scene_name, self.name, i, n_steps, avg_score, round(time.time() - t_start, 3)), logging.INFO)
        # self.save_model(self.arglist)
        return result_scores

    def create_agents(self, arglist):
        _sum = sum(arglist.agent_num)
        if arglist.env == "mpe" or "classic":
            for i in range(_sum):
                if arglist.algorithm == 'ddpg':
                    self.agents.append(MADDPG(arglist, i))
                elif arglist.algorithm == 'td3':
                    self.agents.append(MATD3(arglist, i))
    
    def create_dir(self, arglist):
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
           
    def save_model(self, arglist):
        if not os.path.exists(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        for id, agent in enumerate(self.agents):
            torch.save({'actor_network': agent.actor.state_dict(), 
                    'critic_network': agent.critic.state_dict()}, 
                    os.getcwd() + '/' + self.cur_algo + '_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/' + 
                    str(arglist.scene_name) + '_' +
                    str([arglist.noisy_configuration[i]['distribution']['type'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                    str([arglist.noisy_configuration[i]['distribution']['param'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                    str([arglist.noisy_configuration[i]['reward_influence'] for i in range(len(self.arglist.noisy_configuration))]) +
                    '_agent_' + str(id) + '_model_param.pkl')
        pass