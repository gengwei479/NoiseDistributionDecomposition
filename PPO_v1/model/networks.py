import torch
import torch.nn as nn

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class ActorNetwork_RNN(nn.Module):
    def __init__(self, arglist, actor_input_dim):
        super(ActorNetwork_RNN, self).__init__()
        self.rnn_hidden = None
        self.fc1 = nn.Linear(actor_input_dim, arglist.PPO_v0_rnn_hidden_dim)
        self.rnn = nn.GRUCell(arglist.PPO_v0_rnn_hidden_dim, arglist.PPO_v0_rnn_hidden_dim)
        self.fc2 = nn.Linear(arglist.PPO_v0_rnn_hidden_dim, arglist.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][arglist.PPO_v0_use_relu]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        if arglist.PPO_v0_use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        if torch.is_tensor(actor_input):
            actor_input = actor_input.to(device=self.device)
        else:
            actor_input = torch.tensor(actor_input, device=self.device)
        
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        x = self.fc2(self.rnn_hidden)
        x[avail_a_n == 0] = -1e10
        prob = torch.softmax(x, dim=-1)
        return prob.cpu()


class CriticNetwork_RNN(nn.Module):
    def __init__(self, arglist, critic_input_dim):
        super(CriticNetwork_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, arglist.PPO_v0_rnn_hidden_dim)
        self.rnn = nn.GRUCell(arglist.PPO_v0_rnn_hidden_dim, arglist.PPO_v0_rnn_hidden_dim)
        self.fc2 = nn.Linear(arglist.PPO_v0_rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][arglist.PPO_v0_use_relu]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if arglist.PPO_v0_use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        if torch.is_tensor(critic_input):
            critic_input = critic_input.to(device=self.device)
        else:
            critic_input = torch.tensor(critic_input, device=self.device)
        
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value.cpu()

class VarNetwork_RNN(nn.Module):
    def __init__(self, arglist, critic_input_dim):
        super(VarNetwork_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, arglist.PPO_v0_rnn_hidden_dim)
        self.rnn = nn.GRUCell(arglist.PPO_v0_rnn_hidden_dim, arglist.PPO_v0_rnn_hidden_dim)
        self.fc2 = nn.Linear(arglist.PPO_v0_rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][arglist.PPO_v0_use_relu]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if arglist.PPO_v0_use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        if torch.is_tensor(critic_input):
            critic_input = critic_input.to(device=self.device)
        else:
            critic_input = torch.tensor(critic_input, device=self.device)
        
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value.cpu()

class ActorNetwork_MLP(nn.Module):
    def __init__(self, arglist, actor_input_dim):
        super(ActorNetwork_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, arglist.PPO_v0_mlp_hidden_dim)
        self.fc2 = nn.Linear(arglist.PPO_v0_mlp_hidden_dim, arglist.PPO_v0_mlp_hidden_dim)
        self.fc3 = nn.Linear(arglist.PPO_v0_mlp_hidden_dim, arglist.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][arglist.PPO_v0_use_relu]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        if arglist.PPO_v0_use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        if torch.is_tensor(actor_input):
            actor_input = actor_input.to(device=self.device)
        else:
            actor_input = torch.tensor(actor_input, device=self.device)
        
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        x = self.fc3(x)
        x[avail_a_n == 0] = -1e10
        prob = torch.softmax(x, dim=-1)
        return prob.cpu()


class CriticNetwork_MLP(nn.Module):
    def __init__(self, arglist, critic_input_dim):
        super(CriticNetwork_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, arglist.PPO_v0_mlp_hidden_dim)
        self.fc2 = nn.Linear(arglist.PPO_v0_mlp_hidden_dim, arglist.PPO_v0_mlp_hidden_dim)
        self.fc3 = nn.Linear(arglist.PPO_v0_mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][arglist.PPO_v0_use_relu]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if arglist.PPO_v0_use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        if torch.is_tensor(critic_input):
            critic_input = critic_input.to(device=self.device)
        else:
            critic_input = torch.tensor(critic_input, device=self.device)

        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value.cpu()

class VarNetwork_MLP(nn.Module):
    def __init__(self, arglist, critic_input_dim):
        super(VarNetwork_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, arglist.PPO_v0_mlp_hidden_dim)
        self.fc2 = nn.Linear(arglist.PPO_v0_mlp_hidden_dim, arglist.PPO_v0_mlp_hidden_dim)
        self.fc3 = nn.Linear(arglist.PPO_v0_mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][arglist.PPO_v0_use_relu]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if arglist.PPO_v0_use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        if torch.is_tensor(critic_input):
            critic_input = critic_input.to(device=self.device)
        else:
            critic_input = torch.tensor(critic_input, device=self.device)

        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value.cpu()