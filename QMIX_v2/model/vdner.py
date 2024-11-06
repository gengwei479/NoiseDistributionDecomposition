import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from QMIX_v2.model.utils import init

class Vdner(nn.Module):
    """
    computes Q_tot from individual Q_a values and the state
    """
    def __init__(self, args, device, multidiscrete_list=None):
        """
        init mixer class
        """
        super(Vdner, self).__init__()
        self.device = device
        self.n_agents = args.n_agents       

    def forward(self, agent_q_inps, states):
        """outputs Q_tot, using the individual agent Q values and the centralized env state as inputs"""
        #agent_qs = agent_qs.to(self.device)
        #states = states.to(self.device) 
        batch_size = agent_q_inps.size(0)
        out = agent_q_inps.sum(dim=-1)
        # reshape to (batch_size, 1, 1)
        q_tot = out.view(batch_size, -1, 1)

        #q_tot = q_tot.cpu()
        return q_tot