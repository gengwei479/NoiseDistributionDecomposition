import numpy as np
import random

"""
Class for creating a noise area

type: gauss, uniform, gamma, beta, exponential
params:
    gauss: mu, sigma
    uniform: low, high
    gamma: alpha, beta
    beta: alpha, beta
    exponential: lambd
"""

class NoisyPoint():
    def __init__(self, pos, size, type = 'gauss', reward_influence = 'additive_noise', 
                 influence_weight = 1, params = {}) -> None:
        self.pos = pos
        self.size = size
        self.type = type
        self.params = params
        self.reward_influence = reward_influence
        self.influence_weight = influence_weight
        
        if self.type is 'gauss':
            self.variance = self.params['sigma']
        elif self.type is 'uniform':
            self.variance = (self.params['high'] - self.params['low']) ** 2 / 12
        elif self.type is 'gamma':
            self.variance = self.params['alpha'] / (self.params['beta'] ** 2)
        elif self.type is 'beta':
            self.variance = self.params['alpha'] * self.params['beta'] / ((self.params['alpha'] + self.params['beta'] + 1) * (self.params['alpha'] + self.params['beta']) ** 2)
        elif self.type is 'exponential':
            self.variance = 1 / self.params['lambd'] ** 2
        else:
            self.variance = 0
        pass
    
    def generate_noisy_reward(self, orgin_pos):
        if self.reward_influence is 'additive_noise':
            return self.additive_disturbance(orgin_pos)
        elif self.reward_influence is 'multiplicative_noise':
            return self.multiplicative_disturbance(orgin_pos)
        elif self.reward_influence is 'gmm':
            return self.gmm_disturbance(orgin_pos)
        else:
            return 0

    def additive_disturbance(self, orgin_pos):
        return self.influence_weight * self.state_related_only_disturbance(orgin_pos)
    
    def multiplicative_disturbance(self, orgin_pos):
        return 1 + self.influence_weight * self.state_related_only_disturbance(orgin_pos)
    
    def gmm_disturbance(self, orgin_pos):
        return self.state_related_only_disturbance(orgin_pos)

    def state_related_only_disturbance(self, orgin_pos):
        if self.type is 'gauss' and np.linalg.norm(orgin_pos - self.pos) < self.size:
            return random.gauss(self.params['mu'], self.params['sigma'] ** 0.5)
        elif self.type is 'uniform' and np.linalg.norm(orgin_pos - self.pos) < self.size:
            return random.uniform(self.params['low'], self.params['high'])
        elif self.type is 'gamma' and np.linalg.norm(orgin_pos - self.pos) < self.size:
            return random.gammavariate(self.params['alpha'], self.params['beta'])
        elif self.type is 'beta' and np.linalg.norm(orgin_pos - self.pos) < self.size:
            return random.betavariate(self.params['alpha'], self.params['beta'])
        elif self.type is 'exponential' and np.linalg.norm(orgin_pos - self.pos) < self.size:
            return random.expovariate(self.params['lambd'])
        elif self.type is 'chi' and np.linalg.norm(orgin_pos - self.pos) < self.size:
            return np.random.chisquare(self.params['df'])
        else:
            return 0