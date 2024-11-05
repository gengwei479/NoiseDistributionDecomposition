import numpy as np
import torch

from QMIX_v1.model.networks import Q_network_MLP, Q_network_RNN, QMIX_Net, VDN_Net

class Agent:
    def __init__(self, args):
        self.N = args.agent_num
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.QMIX_add_last_action
        self.add_agent_id = args.QMIX_add_agent_id
        self.max_train_steps=args.QMIX_max_train_steps
        self.lr = args.QMIX_lr
        self.gamma = args.QMIX_gamma
        self.use_grad_clip = args.QMIX_use_grad_clip
        self.batch_size = args.QMIX_batch_size 
        self.target_update_freq = args.QMIX_target_update_freq
        self.tau = args.QMIX_tau
        self.use_hard_update = args.QMIX_use_hard_update
        self.use_rnn = args.QMIX_use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.QMIX_use_double_q
        self.use_RMS = args.QMIX_use_RMS
        self.use_lr_decay = args.QMIX_use_lr_decay

        self.input_dim = self.obs_dim
        if self.add_last_action:
            self.input_dim += self.action_dim
        if self.add_agent_id:
            self.input_dim += self.N

        if self.use_rnn:
            self.eval_Q_net = Q_network_RNN(args, self.input_dim)
            self.target_Q_net = Q_network_RNN(args, self.input_dim)
        else:
            self.eval_Q_net = Q_network_MLP(args, self.input_dim)
            self.target_Q_net = Q_network_MLP(args, self.input_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        if self.algorithm == "qmix":
            self.eval_mix_net = QMIX_Net(args)
            self.target_mix_net = QMIX_Net(args)
        elif self.algorithm == "vdn":
            self.eval_mix_net = VDN_Net()
            self.target_mix_net = VDN_Net()
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        if self.use_RMS:
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)

        self.train_step = 0

    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:
                a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
            else:
                inputs = []
                obs_n = torch.tensor(obs_n, dtype=torch.float32)
                inputs.append(obs_n)
                if self.add_last_action:
                    last_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
                    inputs.append(last_a_n)
                if self.add_agent_id:
                    inputs.append(torch.eye(self.N))

                inputs = torch.cat([x for x in inputs], dim=-1)
                q_value = self.eval_Q_net(inputs)

                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)
                q_value[avail_a_n == 0] = -float('inf')
                a_n = q_value.argmax(dim=-1).numpy()
            return a_n

    def learn(self, replay_buffer, total_steps):
        batch, max_episode_len = replay_buffer.sample()
        self.train_step += 1

        inputs = self.get_inputs(batch, max_episode_len)
        if self.use_rnn:
            self.eval_Q_net.rnn_hidden = None
            self.target_Q_net.rnn_hidden = None
            q_evals, q_targets = [], []
            for t in range(max_episode_len):
                q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.input_dim))
                q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.input_dim))
                q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))
                q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            q_evals = torch.stack(q_evals, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_evals = self.eval_Q_net(inputs[:, :-1])
            q_targets = self.target_Q_net(inputs[:, 1:])

        with torch.no_grad():
            if self.use_double_q:
                q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1)
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]

        q_evals = torch.gather(q_evals, dim=-1, index=batch['a_n'].unsqueeze(-1)).squeeze(-1)

        if self.algorithm == 'qmix':
            q_total_eval = self.eval_mix_net(q_evals, batch['s'][:, :-1])
            q_total_target = self.target_mix_net(q_targets, batch['s'][:, 1:])
        elif self.algorithm == 'vdn':
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)

        targets = batch['r'] + self.gamma * (1 - batch['dw']) * q_total_target

        td_error = (q_total_eval - targets.detach())
        mask_td_error = td_error * batch['active']
        loss = (mask_td_error ** 2).sum() / batch['active'].sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()

        if self.use_hard_update:
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            for param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'])
        if self.add_last_action:
            inputs.append(batch['last_onehot_a_n'])
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len + 1, 1, 1)
            inputs.append(agent_id_one_hot)
        inputs = torch.cat([x for x in inputs], dim=-1)

        return inputs