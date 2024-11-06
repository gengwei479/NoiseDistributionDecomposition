import os
import numpy as np
import random

from NoiseWrapper.NoisyPoint import NoisyPoint

"""
Initialize a noise environment
"""

class NoiseEnv():
    def __init__(self, dim = 2, scene_range = [-1, 1, -1, 1], noise_point_num = 1, 
                 noise_size = [0.2], noise_pos = [], noise_reward_influence = ['additive_noise'], 
                 influence_weights = [], noise_params = []) -> None:
        self.dim = dim
        self.scene_range = scene_range
        self.noise_point_num = noise_point_num        
        self.noise_size = noise_size
        self.noise_pos = noise_pos
        self.noise_reward_influence = noise_reward_influence
        self.influence_weights = influence_weights
        self.influence_weights_sum = [sum(self.influence_weights[: i + 1]) for i in range(len(self.influence_weights))]
        self.noise_params = noise_params
        
        self.noise_entity = []
        
        self.noise_spawn()
        pass
    
    def noise_spawn(self):
        if self.dim == 1:
            range_x_min = self.scene_range[0]
            range_x_max = self.scene_range[1]
            
            for i in range(self.noise_point_num):
                if len(self.noise_pos[i]) is 0:
                    noise_point_positions = np.array([np.random.uniform(range_x_min, range_x_max)])
                else:
                    noise_point_positions = np.array(self.noise_pos[i])
                    
                self.noise_entity.append(NoisyPoint(noise_point_positions, self.noise_size[i], type = self.noise_params[i]['type'], 
                                                    reward_influence = self.noise_reward_influence[i], influence_weight = self.influence_weights[i], 
                                                    params = self.noise_params[i]['param']))
                
        elif self.dim == 2:
            range_x_min = self.scene_range[0]
            range_x_max = self.scene_range[1]
            range_y_min = self.scene_range[2]
            range_y_max = self.scene_range[3]
            
            for i in range(self.noise_point_num):
                if len(self.noise_pos[i]) is 0:
                    noise_point_positions = np.array([np.random.uniform(range_x_min, range_x_max), np.random.uniform(range_y_min, range_y_max)])
                else:
                    noise_point_positions = np.array(self.noise_pos[i])
                
                self.noise_entity.append(NoisyPoint(noise_point_positions, self.noise_size[i], type = self.noise_params[i]['type'], 
                                                    reward_influence = self.noise_reward_influence[i], influence_weight = self.influence_weights[i], 
                                                    params = self.noise_params[i]['param']))
        
        elif self.dim == 3:
            range_x_min = self.scene_range[0]
            range_x_max = self.scene_range[1]
            range_y_min = self.scene_range[2]
            range_y_max = self.scene_range[3]
            range_z_min = self.scene_range[4]
            range_z_max = self.scene_range[5]
            for i in range(self.noise_point_num):
                if len(self.noise_pos[i]) is 0:                
                    noise_point_positions = np.array([np.random.uniform(range_x_min, range_x_max), np.random.uniform(range_y_min, range_y_max), np.random.uniform(range_z_min, range_z_max)])
                else:
                    noise_point_positions = np.array(self.noise_pos[i])
                
                self.noise_entity.append(NoisyPoint(noise_point_positions, self.noise_size[i], type = self.noise_params[i]["type"], 
                                                    reward_influence = self.noise_reward_influence[i], influence_weight = self.influence_weights[i], 
                                                    params = self.noise_params[i]['param']))     

    def noisy_reward(self, positions, orgin_reward):
        total_noise_reward = orgin_reward
        if len(self.noise_entity) != 0 and self.noise_entity[0].reward_influence is 'gmm':
            for pos in positions:
                _rand_int = random.random()
                for id in range(len(self.noise_entity)):
                    if _rand_int <= self.influence_weights_sum[id]:
                        total_noise_reward += self.noise_entity[id].generate_noisy_reward(pos)
                        # print('----********-' + str(id))
                        return total_noise_reward
            
        for item in self.noise_entity:
            for pos in positions:
                if item.reward_influence is 'additive_noise':
                    total_noise_reward += item.generate_noisy_reward(pos)
                elif item.reward_influence is 'multiplicative_noise':
                    total_noise_reward *= item.generate_noisy_reward(pos)
        return total_noise_reward