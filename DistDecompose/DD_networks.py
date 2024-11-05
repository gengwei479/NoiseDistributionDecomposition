import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta, Uniform, Gamma, Exponential, Chi2

class AgentDistribution(nn.Module):
    def __init__(self, input_dim, layer_dim, learning_rate) -> None:
        super(AgentDistribution, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.fc3 = nn.Linear(layer_dim, layer_dim)
        self.mean_layer = nn.Linear(layer_dim, 1)
        self.var_layer = nn.Linear(layer_dim, 1)
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, state):
        if T.is_tensor(state):
            state = T.nn.functional.relu(self.fc1(state.view(1, state.shape[0])))
        else:
            state = T.nn.functional.relu(self.fc1(T.tensor(np.array([state], dtype=np.float32))))
        state = T.nn.functional.relu(self.fc2(state))
        state = T.nn.functional.relu(self.fc3(state))
        mean = self.mean_layer(state)
        var = T.nn.functional.softplus(self.var_layer(state))
        return mean, var
    
    def getdist(self, input):
        mean, var = self.forward(input)
        dist = Normal(mean, var)
        return dist

class TotalDistributionDecompose(nn.Module):
    def __init__(self, input_dim, out_dim, state_dims, layer_dim, learning_rate, reward_range, partition_length, sample_num, episode_num, epsilon, arglist) -> None:# n * N -> n
        super(TotalDistributionDecompose, self).__init__()
        self.q_num = out_dim
        self.agent_num = int(input_dim / out_dim)
        assert len(state_dims) == self.agent_num
        self.weights = nn.Parameter(T.randn(self.agent_num, 1), requires_grad = True)
        self.optimizer = T.optim.Adam(self.parameters(), lr=0.003)
        self.reward_range = reward_range
        self.interval_size = (reward_range[1] - reward_range[0]) / partition_length
        self.reward_values = np.arange(reward_range[0], reward_range[1], (reward_range[1] - reward_range[0]) / partition_length)
        self.reward_pdf = np.zeros(len(self.reward_values))
        self.sample_num = sample_num
        self.episode_num = episode_num
        self.epsilon = epsilon
        self.agentDistribution = [AgentDistribution(state_dims[i], layer_dim, learning_rate) for i in range(self.agent_num)]
        self.arglist = arglist
    
    def kronecker(self, A, B):
        return T.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))
    
    def forward(self, input):
        return T.matmul(input, self.kronecker(nn.functional.softmax(self.weights, dim = 0), T.eye(self.q_num)))
    
    def backpropagation(self, states, dest_dist, noisy_rewards):       
        agent_log_probs = T.tensor([[]])
        for it in range(self.agent_num):
            for r in self.reward_values:
                # print(str(it) + ' aa ' + str(states))
                agent_log_probs = T.cat([agent_log_probs, T.exp(self.agentDistribution[it].getdist(T.tensor(states[it])).log_prob(T.tensor(r))).view(1, 1)], 1)
        
        mean_list = T.tensor([], dtype = T.float)
        for i in range(self.agent_num):
            mean_list = T.cat([mean_list, self.agentDistribution[i].getdist(T.tensor(states[i])).mean], 0)
            
            
        # loss_fun = F.kl_div(dest_dist , varC.forward(agent_log_probs), reduction = 'batchmean')    
        # loss_fun = F.cross_entropy(mean_list.squeeze(), nn.functional.softmax(T.ones(self.agent_num, dtype = T.float)))# + F.cross_entropy(nn.functional.softmax(self.weights, dim=0).squeeze(), nn.functional.softmax(T.ones(self.agent_num)))
        
        _rew_evaluate = sum(noisy_rewards) / len(noisy_rewards)
        loss_fun = F.mse_loss(dest_dist, self.forward(agent_log_probs))# + 100 * F.mse_loss(mean_list.squeeze(), _rew_evaluate * T.ones(self.agent_num, dtype = T.float))
        
        if self.arglist.DD_use_expection_norm:
            loss_fun += self.arglist.DD_lambda * math.pow(10 * math.e, self.arglist.DD_beta - 1) * self.arglist.DD_expection_norm * F.mse_loss(mean_list.squeeze(), _rew_evaluate * T.ones(self.agent_num, dtype = T.float))
        if self.arglist.DD_use_dist_norm:
            loss_fun += self.arglist.DD_lambda * math.log(self.arglist.DD_dist_norm) * T.var(mean_list.squeeze())
        if self.arglist.DD_use_weight_norm:
            loss_fun += self.arglist.DD_beta * math.log(self.arglist.DD_weight_norm) * F.mse_loss(nn.functional.softmax(self.weights, dim=0).squeeze(), nn.functional.softmax(T.ones(self.agent_num), dim=0))
        
        # loss_fun = F.mse_loss(dest_dist, self.forward(agent_log_probs)) + 1 * T.var(mean_list.squeeze())# + 1 * F.mse_loss(nn.functional.softmax(self.weights, dim=0).squeeze(), nn.functional.softmax(T.ones(self.agent_num), dim=0))
        
        # print(loss_fun.item())
        if loss_fun.item() < self.epsilon:
            return False
        
        self.optimizer.zero_grad()
        for i in range(self.agent_num):
            self.agentDistribution[i].optimizer.zero_grad()            
        loss_fun.backward(retain_graph = True)
        self.optimizer.step()
        for i in range(self.agent_num):
            self.agentDistribution[i].optimizer.step()
        return True
    
    def learn_dist(self, cur_states, noisy_rewards):
        self.reward_pdf = np.zeros(len(self.reward_values))
        for _r in range(len(noisy_rewards)):
            self.reward_pdf[int(np.clip(np.round((noisy_rewards[_r] - self.reward_range[0]) / self.interval_size), 0, len(self.reward_values) - 1))] += 1
        self.reward_pdf = T.tensor(np.array([self.reward_pdf / sum(self.reward_pdf)]), dtype=T.float)
        for _ in range(self.episode_num):
            receive_trained = self.backpropagation(cur_states, self.reward_pdf, noisy_rewards)
            # print('a: ' + str(self.episode_num) + ' b: ' + str(self.epsilon))
            if not receive_trained:
                # print('asdsdsd')
                break

        # print('--------' + str(sum(noisy_rewards) / len(noisy_rewards)))
        # print(sorted([self.agentDistribution[i].getdist(cur_states[i]) for i in range(self.agent_num)], key = lambda x: x.mean.item()))
        return sorted([self.agentDistribution[i].getdist(cur_states[i]) for i in range(self.agent_num)], key = lambda x: x.mean.item())