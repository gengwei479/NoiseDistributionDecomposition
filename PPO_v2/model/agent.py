import numpy as np
import torch as T
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from PPO_v2.model.buffer import Buffer
from PPO_v2.model.networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, id, n_actions, input_dims, arglist, device):
        self.id = id
        self.buffer_size = arglist.PPO_buffer_size
        self.batch_size = arglist.PPO_batch_size
        self.max_train_steps = arglist.PPO_max_episodes_num
        self.gamma = arglist.PPO_gamma
        self.lamda = arglist.PPO_lambda
        self.epsilon = arglist.PPO_policy_clip
        self.K_epochs = arglist.PPO_train_times
        self.entropy_coef = arglist.PPO_entropy_coef 
        self.use_grad_clip = True
        self.devide = device
        self.use_adv_norm = True
        self.action_range = arglist.action_range['range']
        
        self.actor = ActorNetwork(input_dims, n_actions, arglist.PPO_actor_learning_rate, arglist.PPO_layer_dim, self.action_range, device)
        self.critic = CriticNetwork(input_dims, arglist.PPO_critic_learning_rate, arglist.PPO_layer_dim, device)
        self.memory = Buffer(self.buffer_size, arglist.state_dim[self.id], arglist.action_dim[self.id])
    
    def remember(self, state, action, probs, reward, state_p, dead_win, done):
        self.memory.store_memory(state, action, probs, reward, state_p, dead_win, done)
      
    def choose_action(self, s):
        s = T.unsqueeze(T.tensor(s, dtype=T.float), 0)            
        with T.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()
                a = T.clamp(a, self.action_range[0], self.action_range[1])
                a_logprob = dist.log_prob(a)
          
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def learn(self):
        s, a, a_logprob, r, s_, dw, done = self.memory.numpy_to_tensor()
        adv = []
        gae = 0
        with T.no_grad():
            vs = self.critic(s.to(self.devide))
            vs_ = self.critic(s_.to(self.devide))
            deltas = r.to(self.devide) + self.gamma * (1.0 - dw.to(self.devide)) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = T.tensor(adv, dtype=T.float).view(-1, 1).to(self.devide)
            v_target = adv + vs
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_size)), self.batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index].to(self.devide))
                ratios = T.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True).to(self.devide))

                surr1 = ratios * adv[index]
                surr2 = T.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -T.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor.optimizer.step()

                v_s = self.critic(s[index].to(self.devide))
                critic_loss = T.nn.functional.mse_loss(v_target[index], v_s)

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic.optimizer.step()
        self.memory.clear_memory()    