import numpy as np
import os
import time
from copy import deepcopy
import torch
import logging
from Common.log_process import Logger


from QMIX_v1.model.normalization import RunningMeanStd, Normalization, RewardScaling
from QMIX_v1.model.buffer import Buffer
from QMIX_v1.model.agent import Agent

class Qmix():
    def __init__(self, name, env, arglist, seed = 0):
        self.name = name
        self.arglist = arglist
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env = env
        
        # Create env
        self.env_info = self.env.get_env_info()
        self.arglist.agent_num = self.env_info["n_agents"]  # The number of agents
        self.arglist.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.arglist.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.arglist.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.arglist.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        
        self.arglist.QMIX_epsilon_decay = (self.arglist.QMIX_epsilon - self.arglist.QMIX_epsilon_min) / self.arglist.QMIX_epsilon_decay_steps

        # Create N agents
        self.agent_n = Agent(self.arglist)
        self.replay_buffer = Buffer(self.arglist)

        self.epsilon = self.arglist.QMIX_epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.arglist.QMIX_use_reward_norm:
            self.reward_norm = Normalization(shape=1)
        
        self.t_start = time.time()
        self.create_dir(arglist) 
        self.logger = Logger(str(self.name) + "_log_info_0000.log", str(os.getcwd()) + '/QMIX' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/')

    def main_run_mpe(self):
        score_history = []
        result_scores = []
        
        learn_iters = 0
        avg_score = 0
        n_steps = 0
        
        for i in range(self.arglist.QMIX_max_train_steps):
            score = 0
            t_start = time.time()
            states = self.env.reset()
            states_prime = {}
            while self.env.agents:
                actions = {}
                probs = {}
                a_n = self.agent_n.choose_action(states)
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
                
                avail_a_n = np.ones([self.arglist.agent_num, ])
                self.replay_buffer.store_transition(i, states, states, avail_a_n, None, a_n, reward, truncations)
                
                states = deepcopy(states_prime)
                states_prime = {}
            score_history.append(score)
            print(str(len(score_history)) + "--------------------")
            if len(score_history) >= 100:
                avg_score = np.mean(score_history[-100:])
                result_scores.append(avg_score)
                print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
                self.logger.log_insert("env:{}, {}, steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(self.arglist.scene_name, self.name, i, n_steps, avg_score, round(time.time() - t_start, 3)), logging.INFO)
        self.save_model(self.arglist)
        return result_scores

    def main_run_smac(self, ):
        evaluate_num = -1  # Record the number of evaluations
        rewards = []
        win_rates = []
        while self.total_steps < self.arglist.QMIX_max_train_steps:
            if self.total_steps // self.arglist.QMIX_evaluate_freq > evaluate_num:
                _rew, _win = self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                rewards.append(_rew)
                win_rates.append(_win)
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.arglist.QMIX_batch_size:
                self.agent_n.learn(self.replay_buffer, self.total_steps)  # Training

        _rew, _win = self.evaluate_policy()
        rewards.append(_rew)
        win_rates.append(_win)
        self.env.close()
        
        return rewards, win_rates

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.arglist.QMIX_evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.arglist.QMIX_evaluate_times
        evaluate_reward = evaluate_reward / self.arglist.QMIX_evaluate_times
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
        if self.arglist.QMIX_use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.arglist.agent_num, self.arglist.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.arglist.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.arglist.action_dim)[a_n]  # Convert actions to one-hot vectors
            r, done, info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.arglist.QMIX_use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.arglist.QMIX_episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.arglist.QMIX_epsilon_decay if self.epsilon - self.arglist.QMIX_epsilon_decay > self.arglist.QMIX_epsilon_min else self.arglist.QMIX_epsilon_min

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)

        return win_tag, episode_reward, episode_step + 1
    
    def create_dir(self, arglist):
        if not os.path.exists(os.getcwd() + '/QMIX' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/QMIX' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/QMIX' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/QMIX' + str(arglist.log_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
        if not os.path.exists(os.getcwd() + '/QMIX' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/QMIX' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name))
            
    def save_model(self, arglist):
        if not os.path.exists(os.getcwd() + '/QMIX' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name)):
            os.makedirs(os.getcwd() + '/QMIX' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name))

        torch.save({'q_network': self.agent_n.target_Q_net.state_dict(), 
                'mix_network': self.agent_n.target_mix_net.state_dict()}, 
                os.getcwd() + '/QMIX' + str(arglist.model_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/' + 
                str(arglist.scene_name) + '_' +
                str([arglist.noisy_configuration[i]['distribution']['type'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                str([arglist.noisy_configuration[i]['distribution']['param'] for i in range(len(self.arglist.noisy_configuration))]) + '_' +
                str([arglist.noisy_configuration[i]['reward_influence'] for i in range(len(self.arglist.noisy_configuration))]) +
                '_agent_' + str(id) + '_model_param.pkl')
        pass