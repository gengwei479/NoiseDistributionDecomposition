import numpy as np
import torch
import gym
# from gym.spaces import Box, Discrete, Tuple
from gym.spaces import Tuple
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def make_onehot(ind_vector, batch_size, action_dim, seq_len=None):
    if not seq_len:
        onehot_mat = torch.zeros((batch_size, action_dim)).float()
        onehot_mat[torch.arange(batch_size), ind_vector.long()] = 1
        return onehot_mat
    if seq_len:
        onehot_mats = []
        for i in range(seq_len):
            mat = torch.zeros((batch_size, action_dim)).float()
            mat[torch.arange(batch_size), ind_vector[i].long()] = 1
            onehot_mats.append(mat)
        return torch.stack(onehot_mats)

class DecayThenFlatSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, Discrete):
        dim = space.n
    elif isinstance(space, Tuple):
        dim = sum([get_dim_from_space(sp) for sp in space])
    elif isinstance(space, MultiDiscrete):
        # TODO: support multidiscrete spaces
        return (space.high - space.low) + 1

    elif isinstance(space, list):
        dim = space[0]
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim

def get_state_dim(observation_dict, action_dict):
    combined_obs_dim = sum([get_dim_from_space(space) for space in list(observation_dict.values())])
    combined_act_dim = 0
    for space in list(action_dict.values()):
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            combined_act_dim += int(sum(dim))
        else:
            combined_act_dim += dim
    return combined_obs_dim, combined_act_dim, combined_obs_dim+combined_act_dim

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample()

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

def avail_choose(x, available_actions=None):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    if type(available_actions) == np.ndarray:
        available_actions = torch.from_numpy(available_actions)

    x[available_actions==0]=-1e10
    return FixedCategorical(logits=x)

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]
    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)