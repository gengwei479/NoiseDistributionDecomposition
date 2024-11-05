import numpy as np
from NoiseWrapper.NoisyScene import NoiseEnv

class NoiseWrapperPAR():
    def __init__(self, env_type_name, base_env, dim, partial_state, scene_range, noisy_configuration, cooperative_agents) -> None:
        self.env_type_name = env_type_name
        self.base_env = base_env
        self.cur_pos = None
        self.partial_state = partial_state
        
        self.cooperative_agents = cooperative_agents
        self.cooperative_agents_id = []
        
        if self.env_type_name == "mpe":
            # self.observation_spaces = self.base_env.observation_spaces
            self.num_agents = self.base_env.num_agents
            self.agents = self.base_env.agents
            
            for id in range(len(self.agents)):
                if self.cooperative_agents[id]:
                    self.cooperative_agents_id.append(id)
        elif self.env_type_name == "smac":
            self.num_agents = self.get_env_info()["n_agents"]
            
            for id in range(self.num_agents):
                if self.cooperative_agents[id]:
                    self.cooperative_agents_id.append(id)
        
        
        self.noisy_environ = NoiseEnv(dim, scene_range, len(noisy_configuration), [i["size"] for i in noisy_configuration], 
                                      [i["pos"] for i in noisy_configuration], 
                                      [i["reward_influence"] for i in noisy_configuration], 
                                      [i["reward_influence_weight"] for i in noisy_configuration], 
                                      [i["distribution"] for i in noisy_configuration])
            
    
    def reset(self):        
        if self.env_type_name == "mpe":
            states = self.base_env.reset()
            self.num_agents = self.base_env.num_agents
            self.agents = self.base_env.agents
            self.cur_pos = [states[self.base_env.agents[i]][self.partial_state[0]:] if self.partial_state[1] == 0 else states[self.base_env.agents[i]][self.partial_state[0]:self.partial_state[1]] 
                            for i in range(len(self.agents))]
            return states
        elif self.env_type_name == "smac":
            self.base_env.reset()
            states = self.get_obs()
            self.num_agents = self.base_env.get_env_info()["n_agents"]
            self.cur_pos = [states[i][self.partial_state[0]:] if self.partial_state[1] == 0 else states[i][self.partial_state[0]:self.partial_state[1]] 
                for i in range(self.num_agents)]
    
    def step(self, actions):
        if self.env_type_name == "mpe":
            self.num_agents = self.base_env.num_agents
            self.agents = self.base_env.agents
            states_prime, rewards, terminations, truncations, infos = self.base_env.step(actions)
            self.cur_pos = [states_prime[self.base_env.agents[i]][self.partial_state[0]:] if self.partial_state[1] == 0 else states_prime[self.base_env.agents[i]][self.partial_state[0]:self.partial_state[1]] 
                            for i in range(len(self.agents))]
            
            noisy_rewards = {}
            for id in range(len(self.agents)):
                if self.cooperative_agents[id]:
                    noisy_rewards[self.agents[id]] = self.noisy_environ.noisy_reward(self.cur_pos, rewards[self.agents[id]])
                else:
                    noisy_rewards[self.agents[id]] = rewards[self.agents[id]]
            return states_prime, noisy_rewards, terminations, truncations, infos
        
        elif self.env_type_name == "smac":
            self.num_agents = self.base_env.get_env_info()["n_agents"]
            reward, done, info = self.base_env.step(actions)
            states_prime = self.get_obs()
            self.cur_pos = [states_prime[i][self.partial_state[0]:] if self.partial_state[1] == 0 else states_prime[i][self.partial_state[0]:self.partial_state[1]] 
                for i in range(self.num_agents)]

            noisy_reward = self.noisy_environ.noisy_reward(self.cur_pos, reward)
            return noisy_reward, done, info       
            
            
    def step_sample(self, actions, sample_num):   
        if self.env_type_name == "mpe":
            self.num_agents = self.base_env.num_agents
            self.agents = self.base_env.agents
            states_prime, rewards, terminations, truncations, infos = self.base_env.step(actions)
            self.cur_pos = [states_prime[self.base_env.agents[i]][self.partial_state[0]:] if self.partial_state[1] == 0 else states_prime[self.base_env.agents[i]][self.partial_state[0]:self.partial_state[1]] 
                for i in range(len(self.agents))]
            noisy_rewards = []
            for _ in range(sample_num):
                noisy_rewards.append(self.noisy_environ.noisy_reward(self.cur_pos, list(rewards.values())[self.cooperative_agents_id[0]]))
            
            res_rewards = [noisy_rewards, rewards]
            return states_prime, res_rewards, terminations, truncations, infos
        
        elif self.env_type_name == "smac":
            self.num_agents = self.base_env.get_env_info()["n_agents"]
            reward, done, info = self.base_env.step(actions)
            states_prime = self.get_obs()
            self.cur_pos = [states_prime[i][self.partial_state[0]:] if self.partial_state[1] == 0 else states_prime[i][self.partial_state[0]:self.partial_state[1]] 
                for i in range(self.num_agents)]
            noisy_rewards = []
            for _ in range(sample_num):
                noisy_rewards.append(self.noisy_environ.noisy_reward(self.cur_pos, reward))
            # print('pre ' + str(sum(noisy_rewards) / len(noisy_rewards)) + '  org: ' + str(reward))
            res_rewards = [noisy_rewards, reward]
            return res_rewards, done, info  
    
    def close(self):
        if self.env_type_name == "smac":
            self.base_env.close()
    
    def get_env_info(self):
        if self.env_type_name == "smac":
            return self.base_env.get_env_info()
    
    def get_obs(self):
        if self.env_type_name == "smac":
            return self.base_env.get_obs()
    
    def get_state(self):
        if self.env_type_name == "smac":
            return self.base_env.get_state()
    
    def get_avail_actions(self):
        if self.env_type_name == "smac":
            return self.base_env.get_avail_actions()