# from gym.spaces import Box, Discrete
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box
from QMIX_v2.model.utils import MultiDiscrete
from QMIX_v2.model.multiagent_particle_envs.mpe.make_env import make_env
from QMIX_v2.model.multiagent_particle_envs.vec_env_wrappers import DummyVecEnv
import numpy as np
from NoiseWrapper.NoisyScene import NoiseEnv
import gym.spaces.tuple as tuple

class ParticleEnvMultiEnv():
    def __init__(self, arglist, is_noise = True):
        self.is_noise = is_noise
        self.partial_state = arglist.partial_observation
        
        # self._env = make_env(arglist.scene_name)
        # self._env = make_env('simple_speaker_listener')
    
        from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_speaker_listener_v3, simple_spread_v2, simple_reference_v2
    #     arglist.env_continuous = True
        _env = None
        if arglist.scene_name == "simple_adversary":
            _env = simple_adversary_v2.parallel_env(continuous_actions = False)
            arglist.cooperative_agents = [False, True, True]
        elif arglist.scene_name == "simple_crypto":
            _env = simple_crypto_v2.parallel_env(continuous_actions = False)
            arglist.cooperative_agents = [False, True, True]
        elif arglist.scene_name == "simple_speaker_listener":
            _env = simple_speaker_listener_v3.parallel_env(continuous_actions = False)
            arglist.cooperative_agents = [True, True]
        elif arglist.scene_name == "simple_spread":
            _env = simple_spread_v2.parallel_env(continuous_actions = False)
            arglist.cooperative_agents = [True, True, True]
        elif arglist.scene_name == "simple_reference":
            _env = simple_reference_v2.parallel_env(local_ratio=0, continuous_actions = False)
            arglist.cooperative_agents = [True, True]
        self._env = _env
        
        self._env.reset()
        self.agent_ids = [i for i in range(self._env.num_agents)]
        self.num_agents = self._env.num_agents
        self.agents = self._env.agents
        self.observation_space_dict = self._convert_to_dict(list(self._env.observation_spaces.values()))
        self.action_space_dict = self._convert_to_dict(list(self._env.action_spaces.values()))
        
        self.cooperative_agents = arglist.cooperative_agents
        noisy_configuration = arglist.noisy_configuration
        self.noisy_environ = NoiseEnv(arglist.dim, arglist.scene_range, len(noisy_configuration), [i["size"] for i in noisy_configuration], 
                                [i["pos"] for i in noisy_configuration], 
                                [i["reward_influence"] for i in noisy_configuration], 
                                [i["reward_influence_weight"] for i in noisy_configuration], 
                                [i["distribution"] for i in noisy_configuration])

    def reset(self):
        return self._convert_to_dict(list(self._env.reset().values()))

    def step(self, action_dict):
        action_dict_modify = {}
        for id in action_dict.keys():
            action = action_dict[id]
            # action_space = self.action_space_dict[id]
            # converted_action = self._convert_action(action_space, action)
            # action_dict_modify[self.agents[id]] = converted_action
            
            action_dict_modify[self.agents[id]] = np.where(action)[0][0]

        obs_dic, rew_dic, done_dic, _, _ = self._env.step(action_dict_modify)
        
        obs_list, rew_list, done_list = list(obs_dic.values()), list(rew_dic.values()), list(done_dic.values())
        
        if self.is_noise:
            noisy_rewards = []
            for id in range(self.num_agents):
                self.cur_pos = [obs_list[id][self.partial_state[0]:] if self.partial_state[1] == 0 else obs_list[id][self.partial_state[0]:self.partial_state[1]] 
                for i in range(len(self.agents))]
                
                if self.cooperative_agents[id]:
                    noisy_rewards.append(self.noisy_environ.noisy_reward(self.cur_pos, rew_list[id]))
                else:
                    noisy_rewards.append(rew_list[id])
            rewards = self._convert_to_dict(noisy_rewards)
        else:
            rewards = self._convert_to_dict(rew_list)

        obs = self._convert_to_dict(obs_list)
        dones = self._convert_to_dict(done_list)
        dones['env'] = all([dones[agent_id] for agent_id in self.agent_ids])
        infos = self._convert_to_dict([{"done": done} for done in done_list])

        return obs, rewards, dones, infos

    def seed(self, seed):
        self._env.seed(seed)

    def render(self, mode='human'):
        self._env.render(mode=mode)

    def _convert_to_dict(self, vals):
        """
        Convert a list of per-agent values into a dict mapping agent_id to the agent's corresponding value.
        Args:
            vals: list of per-agent values. Must be of length self.num_agents
        Returns:
            dict: dictionary mapping agent_id to the agent' corresponding value, as specified in vals
        """
        return dict(zip(self.agent_ids, vals))

    def _convert_action(self, action_space, action):
        # print(type(action_space))
        # print(action)
        if isinstance(action_space, Discrete):
            if type(action) == np.ndarray and len(action) == action_space.n:
                converted_action = action
            else:
                converted_action = np.zeros(action_space.n)
                if type(action) == np.ndarray or type(action) == list:
                    converted_action[action[0]] = 1.0
                else:
                    converted_action[action] = 1.0
        elif isinstance(action_space, Box):
            converted_action = action

        elif isinstance(action_space, MultiDiscrete):
            if type(action) == list:
                action = np.concatenate(action)
            total_dim = sum((action_space.high - action_space.low) + 1)
            assert type(action) == np.ndarray and len(action) == total_dim, "Invalid MultiDiscrete action!"
            return action
        # elif isinstance(action_space, tuple.Tuple):
        #     if type(action) == np.ndarray:
        #         return np.array(action)
        else:
            raise Exception("Unsupported space")

        return converted_action


def make_parallel_env(arglist, is_noise):
    def get_env_fn():
        def init_env():
            # if arglist.scene_name == "simple_adversary":
            #     arglist.cooperative_agents = [False, True, True]
            # elif arglist.scene_name == "simple_crypto":
            #     arglist.cooperative_agents = [False, True, True]
            # elif arglist.scene_name == "simple_speaker_listener":
            #     arglist.cooperative_agents = [True, True]
            # elif arglist.scene_name == "simple_spread":
            #     arglist.cooperative_agents = [True, True, True]
            # elif arglist.scene_name == "simple_reference":
            #     arglist.cooperative_agents = [True, True]
            env = ParticleEnvMultiEnv(arglist, is_noise)
            return env
        return init_env

    return DummyVecEnv([get_env_fn()])


# def make_parallel_env(arglist):
#     
#     return _env