import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import matplotlib.pyplot as plt

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, input_dim, num_units, depth, n_steps = 10):
        super(MLPDiffusion, self).__init__()
        self.depth = depth

        # self.linears = nn.ModuleList(
        #     [
        #         nn.Linear(input_dim, num_units),
        #         nn.ReLU(),
        #         nn.Linear(num_units, num_units),
        #         nn.ReLU(),
        #         nn.Linear(num_units, num_units),
        #         nn.ReLU(),
        #         nn.Linear(num_units, input_dim),
        #     ]
        # )
        # self.step_embeddings = nn.ModuleList(
        #     [
        #         nn.Embedding(n_steps, num_units),
        #         nn.Embedding(n_steps, num_units),
        #         nn.Embedding(n_steps, num_units),
        #     ]
        # )
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, num_units), nn.ReLU()
            ] + 
            [                
                x for y in zip([nn.Linear(num_units, num_units) for i in range(depth - 1)], [nn.ReLU() for i in range(depth - 1)]) for x in y
            ] + 
            [
                nn.Linear(num_units, input_dim)
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units) for i in range(depth)
            ]
        )

    def forward(self, x):
        #         x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(torch.tensor([self.depth]))
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x
    
class DiffusionGD:
    def __init__(self, arglist) -> None:
        self.arglist = arglist
        self.input_dim = self.arglist.DM_input_dim
        self.model = MLPDiffusion(self.input_dim, self.arglist.DM_agent_network_layer_dim, self.arglist.DM_diffusion_depth)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.arglist.DM_learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def learn(self, target_samples):
        for epoch in range(self.arglist.DM_num_epochs):
            # 生成噪声样本
            noise_samples = torch.randn([1, self.input_dim])

            # 通过扩散模型生成样本
            generated_samples = self.model(noise_samples)

            # 计算损失
            loss = self.loss_fn(generated_samples, torch.tensor([target_samples]))

            # 反向传播及优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 打印损失
            # if (epoch+1) % 10 == 0:
            #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.arglist.DM_num_epochs, loss.item()))
    
    def dg(self, target_samples, generate_size):
        self.learn(target_samples)
        
        generated_samples = []
        while len(generated_samples) < self.arglist.DD_reward_sample_num:
            noise_samples = torch.randn([1, self.input_dim])
            generated_samples += self.model(noise_samples).squeeze().tolist()
        return generated_samples

# parser = argparse.ArgumentParser("param configuration")
# parser.add_argument("--DM_agent_network_layer_dim", type = int, default = 128, help = "units number of dm-networks")
# parser.add_argument("--DM_diffusion_depth", type = int, default = 3, help = "diffusion times of dm-networks")
# parser.add_argument("--DM_learning_rate", type = float, default = 0.001, help = "learning rate of dm-networks")
# parser.add_argument("--DM_num_epochs", type = int, default = 1000, help = "update times of dm-networks")
# parser.add_argument("--DM_input_dim", type = int, default = 50, help = "input dim of dm-networks")
# parser.add_argument("--DD_reward_sample_num", type = int, default = 500, help = "sample times for reward noise")

# arglist = parser.parse_args()


# target_distribution0 = torch.distributions.Normal(torch.tensor([20.0] * arglist.DM_input_dim), torch.tensor([5.0] * arglist.DM_input_dim))
# target_distribution1 = torch.distributions.Normal(torch.tensor([0.0] * arglist.DM_input_dim), torch.tensor([5.0] * arglist.DM_input_dim))
# target_samples = (0.35 * target_distribution0.sample() + 0.65 * target_distribution1.sample()).reshape(1, arglist.DM_input_dim).tolist()

# # target_distribution = torch.distributions.Normal(torch.tensor([20.0] * arglist.DM_input_dim), torch.tensor([1.0] * arglist.DM_input_dim))
# # target_samples = target_distribution.sample().reshape(1, arglist.DM_input_dim).tolist()

# dGD = DiffusionGD(arglist)
# generated_samples = dGD.dg(target_samples, 500)
# plt.hist(generated_samples, 100, color ='green',alpha = 0.7, density = True)
# plt.show()
# print(sum(generated_samples) / len(generated_samples))