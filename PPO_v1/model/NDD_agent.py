import numpy as np
import copy
import torch
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from PPO_v1.model.networks import ActorNetwork_MLP, ActorNetwork_RNN, CriticNetwork_MLP, CriticNetwork_RNN, VarNetwork_MLP, VarNetwork_RNN

class Agent:
    def __init__(self, arglist):
        self.N = arglist.N
        self.obs_dim = arglist.obs_dim
        self.state_dim = arglist.state_dim
        self.action_dim = arglist.action_dim

        self.is_global = arglist.PPO_v0_use_global_state
        self.batch_size = arglist.PPO_v0_batch_size
        self.mini_batch_size = arglist.PPO_v0_mini_batch_size
        self.max_train_steps = arglist.PPO_v0_max_train_steps
        self.lr = arglist.PPO_v0_lr
        self.gamma = arglist.PPO_v0_gamma
        self.lamda = arglist.PPO_v0_lamda
        self.epsilon = arglist.PPO_v0_epsilon
        self.K_epochs = arglist.PPO_v0_K_epochs
        self.entropy_coef = arglist.PPO_v0_entropy_coef
        self.set_adam_eps = arglist.PPO_v0_set_adam_eps
        self.use_grad_clip = arglist.PPO_v0_use_grad_clip
        self.use_lr_decay = arglist.PPO_v0_use_lr_decay
        self.use_adv_norm = arglist.PPO_v0_use_adv_norm
        self.use_rnn = arglist.PPO_v0_use_rnn
        self.add_agent_id = arglist.PPO_v0_add_agent_id
        self.use_agent_specific = arglist.PPO_v0_use_agent_specific
        self.use_value_clip = arglist.PPO_v0_use_value_clip
        self.actor_input_dim = arglist.obs_dim
        if self.is_global:
            self.critic_input_dim = arglist.state_dim
        else:
            self.critic_input_dim = 0
        if self.add_agent_id:
            self.actor_input_dim += arglist.N
            self.critic_input_dim += arglist.N
        if self.use_agent_specific:
            self.critic_input_dim += arglist.obs_dim

        if self.use_rnn:
            self.actor = ActorNetwork_RNN(arglist, self.actor_input_dim)
            self.critic = CriticNetwork_RNN(arglist, self.critic_input_dim)
            self.var = VarNetwork_RNN(arglist, self.critic_input_dim)
        else:
            self.actor = ActorNetwork_MLP(arglist, self.actor_input_dim)
            self.critic = CriticNetwork_MLP(arglist, self.critic_input_dim)
            self.var = VarNetwork_MLP(arglist, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.var.parameters())
        if self.set_adam_eps:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, avail_a_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N))

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)
            prob = self.actor(actor_inputs, avail_a_n)
            if evaluate:
                a_n = prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s, obs_n):
        with torch.no_grad():
            critic_inputs = []
            if self.is_global:
                s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
                critic_inputs.append(s)
            if self.use_agent_specific:
                critic_inputs.append(torch.tensor(obs_n, dtype=torch.float32))
            if self.add_agent_id:
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
            v_n = self.critic(critic_inputs)
            var_n = self.var(critic_inputs)
            return v_n.numpy().flatten(), var_n.numpy().flatten()

    def learn(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()
        max_episode_len = replay_buffer.max_episode_len

        adv = []
        gae = 0
        with torch.no_grad():
            deltas = batch['r'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['dw']) - batch['v_n'][:, :-1]
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v_n'][:, :-1]
            var_target = batch['var'] + pow(self.gamma, 2) * batch['var_n'][:, 1:] * (1 - batch['dw'])
            
            if self.use_adv_norm:
                adv_copy = copy.deepcopy(adv.numpy())
                adv_copy[batch['active'].numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))

        actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)

        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                if self.use_rnn:
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    self.var.rnn_hidden = None
                    probs_now, values_now, vars_now = [], [], []
                    for t in range(max_episode_len):
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1),
                                          batch['avail_a_n'][index, t].reshape(self.mini_batch_size * self.N, -1))
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                        values_now.append(v.reshape(self.mini_batch_size, self.N))
                        var = self.var(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                        vars_now.append(var.reshape(self.mini_batch_size, self.N))
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                    vars_now = torch.stack(vars_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index], batch['avail_a_n'][index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)
                    vars_now = self.var(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                #----------
                if self.use_value_clip:
                    vars_old = batch["var_n"][index, :-1].detach()
                    vars_error_clip = torch.clamp(vars_now - vars_old, -self.epsilon, self.epsilon) + vars_old - var_target[index]
                    vars_error_original = vars_now - var_target[index]
                    var_loss = torch.max(vars_error_clip ** 2, vars_error_original ** 2)
                else:
                    var_loss = (vars_now - var_target[index]) ** 2
                var_loss = (var_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                #----------

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss + var_loss
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        if self.is_global:
            critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.use_agent_specific:
            critic_inputs.append(batch['obs_n'])
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        return actor_inputs, critic_inputs

    # def save_model(self, env_name, number, seed, total_steps):
    #     torch.save(self.actor.state_dict(), "./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    # def load_model(self, env_name, number, seed, step):
    #     self.actor.load_state_dict(torch.load("./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))