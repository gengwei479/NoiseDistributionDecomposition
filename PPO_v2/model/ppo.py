import gym
import os
import time
import numpy as np
import torch as T
import random
from copy import deepcopy

from PPO_v2.model.normalization import Normalization
from PPO_v2.model.agent import Agent
import logging
from Common.log_process import Logger

class Ppo_v2():
    def __init__(self, name,  env, arglist, seed = 0) -> None:
        self.name = name
        self.seed = seed
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        
        self.agents = []
        self.arglist = arglist
        self.nn_device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.nn_device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.env = env
        # self.test_env = test_env
        self.env.reset()
        
        self.max_episodes_num = arglist.PPO_max_episodes_num
        self.save_rate = arglist.PPO_save_rate
                
        self.create_agents(self.arglist)
        self.state_norm = [Normalization(shape = arglist.state_dim) for i in range(len(self.agents))]
        self.reward_norm = [Normalization(shape = 1) for _ in range(len(self.agents))]
        
        self.create_dir(arglist)
        self.logger = Logger(str(self.name) + "_log_info_0000.log", str(os.getcwd()) + '/PPO_v2' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/')
        pass

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
                probs = {}
                for agent_id in range(self.env.num_agents):
                    _action, _prob = self.agents[agent_id].choose_action(states[self.env.agents[agent_id]])
                    actions[self.env.agents[agent_id]] = _action
                    probs[self.env.agents[agent_id]] = _prob
                states_prime, rewards, terminations, truncations, infos = self.env.step(actions)
                
                ## reward will be equal for every agent because of their shared rewards.
                if len(self.env.agents) != 0:
                    reward = sum(list(rewards.values()))
                
                n_steps += 1
                score += sum(list(rewards.values()))
                
                for agent_id in range(self.env.num_agents):
                    if len(states) != 0 and len(actions) != 0 and len(rewards) != 0 and len(states_prime) != 0:
                        self.agents[agent_id].remember(states[self.env.agents[agent_id]], actions[self.env.agents[agent_id]], probs[self.env.agents[agent_id]], 
                                                       rewards[self.env.agents[agent_id]], states_prime[self.env.agents[agent_id]], False, truncations[self.env.agents[agent_id]])
                    if self.agents[agent_id].memory.count == self.arglist.PPO_buffer_size:
                        self.agents[agent_id].learn()
                        learn_iters += 1
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
                self.agents.append(Agent(i, input_dims = arglist.state_dim[i], n_actions = arglist.action_dim[i], arglist = arglist, device = self.nn_device))
    
    def create_dir(self, arglist):
        if not os.path.exists(os.getcwd() + '/PPO_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/PPO_v2' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v2' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/PPO_v2' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v2' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
           
    def save_model(self, arglist):
        if not os.path.exists(os.getcwd() + '/PPO_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/PPO_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        for id, agent in enumerate(self.agents):
            T.save({'actor_network': agent.actor.state_dict(), 
                    'critic_network': agent.critic.state_dict()}, 
                    os.getcwd() + '/PPO_v2' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/' + 
                    str(arglist.scene_name) + '_' +
                    str([arglist.noisy_configuration[i]['distribution']['type'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                    str([arglist.noisy_configuration[i]['distribution']['param'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                    str([arglist.noisy_configuration[i]['reward_influence'] for i in range(len(self.arglist.noisy_configuration))]) +
                    '_agent_' + str(id) + '_model_param.pkl')
        pass