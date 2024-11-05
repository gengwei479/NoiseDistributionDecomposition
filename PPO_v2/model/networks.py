import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal, Beta

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, alpha, layer_dim, action_range, device):
        super(ActorNetwork, self).__init__()
        self.action_range = action_range        
        self.fc1 = nn.Linear(input_dims, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.alpha_layer = nn.Linear(layer_dim, output_dims)
        self.beta_layer = nn.Linear(layer_dim, output_dims)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, s):        
        s = T.nn.functional.relu(self.fc1(s.to(self.device)))
        s = T.nn.functional.relu(self.fc2(s))
        alpha = T.nn.functional.softplus(self.alpha_layer(s)) + 1.0
        beta = T.nn.functional.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):        
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist
    
    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)
        return mean

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, layer_dim, device):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.fc3 = nn.Linear(layer_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, s):
        s = T.nn.functional.relu(self.fc1(s))
        s = T.nn.functional.relu(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

class VarNetwork(nn.Module):
    def __init__(self, input_dims, alpha, layer_dim, device):
        super(VarNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.fc3 = nn.Linear(layer_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, s):
        s = T.nn.functional.relu(self.fc1(s))
        s = T.nn.functional.relu(self.fc2(s))
        v = T.exp(self.fc3(s))
        return v