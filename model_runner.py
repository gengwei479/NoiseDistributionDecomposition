import os
import numpy as np
import arg_config as arg_config
import visual_resault

from PPO_v1.model.ppo import Ppo_v1
from PPO_v1.model.NDD_ppo import NDD_Ppo_v1
from PPO_v2.model.ppo import Ppo_v2
from PPO_v2.model.NDD_ppo import NDD_Ppo_v2

from QMIX_v2.model.qmix import Qmix
# from QMIX_v2.model.multiagent_particle_envs.particle_env_multienv import make_parallel_env
from QMIX_v2.model.mpe_wrap import make_parallel_env

from DDPG.model.ddpg import Ddpg_v1, Ddpg_v2

from NoiseWrapper.NoiseEnvWrapper import NoiseWrapperPAR

def make_env_MPE_v0(arglist, is_noise):
    arglist.dim = 2
    arglist.scene_range = [-1, 1, -1, 1]
    arglist.partial_observation = [-arglist.dim, 0]
    env = make_parallel_env(arglist, is_noise)

    env.reset()
    arglist.n_agents = env.num_agents
    arglist.agent_ids = [i for i in range(env.num_agents)]
    # arglist.state_dim = [env.observation_space(_agent).shape[0] for _agent in env.agents]
    # arglist.action_dim = [env.action_space(_agent).shape[0] for _agent in env.agents]
    # arglist.action_range = {'shape': [5], 'range': [0, 1]}
    arglist.reward_range = [-10, 10]
    return env

def make_env_MPE(arglist):
    from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_speaker_listener_v3, simple_spread_v2, simple_reference_v2
    arglist.env_continuous = True
    env = None
    if arglist.scene_name == "simple_adversary":
        env = simple_adversary_v2.parallel_env(continuous_actions = True)
        arglist.cooperative_agents = [False, True, True]
    elif arglist.scene_name == "simple_crypto":
        env = simple_crypto_v2.parallel_env(continuous_actions = True)
        arglist.cooperative_agents = [False, True, True]
    elif arglist.scene_name == "simple_speaker_listener":
        env = simple_speaker_listener_v3.parallel_env(continuous_actions = True)
        arglist.cooperative_agents = [True, True]
    elif arglist.scene_name == "simple_spread":
        env = simple_spread_v2.parallel_env(continuous_actions = True)
        arglist.cooperative_agents = [True, True, True]
    elif arglist.scene_name == "simple_reference":
        env = simple_reference_v2.parallel_env(local_ratio=0, continuous_actions = True)
        arglist.cooperative_agents = [True, True]

    assert env is not None
    
    env.reset()
    arglist.dim = 2
    arglist.scene_range = [-1, 1, -1, 1]
    arglist.agent_num = [1 for _ in env.agents]
    arglist.state_dim = [env.observation_space(_agent).shape[0] for _agent in env.agents]
    arglist.action_dim = [env.action_space(_agent).shape[0] for _agent in env.agents]
    arglist.action_range = {'shape': [5], 'range': [0, 1]}
    arglist.partial_observation = [-arglist.dim, 0]
    arglist.reward_range = [-10, 10]
    return env

def make_env_SMAC(arglist):
    arglist.env_continuous = False
    from smac.env.starcraft2.starcraft2  import StarCraft2Env
    env = StarCraft2Env(map_name = arglist.scene_name, seed = 0)
    env.reset()
      
    arglist.dim = 2
    arglist.scene_range = [0, env.map_x, 0, env.map_y]
    
    env_info = env.get_env_info()
    arglist.state_dim = [env_info["obs_shape"]] * env_info["n_agents"]
    arglist.action_dim = [env_info["n_actions"]] * env_info["n_agents"]
    arglist.action_range = {'shape': env_info["n_actions"], 'range': [0, 1]}
    
    arglist.partial_observation = [-arglist.dim, 0]
    arglist.reward_range = [-5, 25]
    arglist.cooperative_agents = [True for _ in range(env.get_env_info()["n_agents"])]
    return env

def make_env_MPE_v0_noise(arglist, is_noise):
    _env = make_env_MPE_v0(arglist, is_noise)
    # _env_wrapper = NoiseWrapperPAR('mpe', _env, arglist.dim, arglist.partial_observation, arglist.scene_range, arglist.noisy_configuration, arglist.cooperative_agents)
    return _env

def make_env_MPE_noise(arglist):
    _env = make_env_MPE(arglist)
    _env_wrapper = NoiseWrapperPAR('mpe', _env, arglist.dim, arglist.partial_observation, arglist.scene_range, arglist.noisy_configuration, arglist.cooperative_agents)
    return _env_wrapper

def make_env_SMAC_noise(arglist):
    _env = make_env_SMAC(arglist)
    _env_wrapper = NoiseWrapperPAR('smac', _env, arglist.dim, arglist.partial_observation, arglist.scene_range, arglist.noisy_configuration, arglist.cooperative_agents)
    return _env_wrapper

def ppo_main_run(arglist):
    arglist.noisy_configuration = eval(arglist.noisy_configuration)
    arglist.scene_name = str(arglist.scene_name)
    
    if arglist.env == "mpe":        
        # base_env = make_env_MPE(arglist)
        # noise_env = make_env_MPE_noise(arglist)
        noise_env_c = make_env_MPE_noise(arglist)
        
        # rewards = []
        # trained_rewards = []
        ndd_trained_rewards = []
        
        # if os.path.exists(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_ndd_rew_lambda_0_beta_0_risk' + str(arglist.DD_risk_sensitive) + str(arglist.DD_eta) + '_DMnum' + str(arglist.DM_num_epochs / arglist.DD_reward_sample_num) + '.npy'):
        #     print(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_ndd_rew_lambda_0_beta_0_risk' + str(arglist.DD_risk_sensitive) + str(arglist.DD_eta) + '_DMnum' + str(arglist.DM_num_epochs / arglist.DD_reward_sample_num) + '.npy')
        #     print('--exist')
        #     return
        
        # print(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_ndd_rew_lambda_0_beta_0_risk' + str(arglist.DD_risk_sensitive) + str(arglist.DD_eta) + '_DMnum' + str(arglist.DM_num_epochs / arglist.DD_reward_sample_num) + '.npy')
        # print('--nope')
        print(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_ndd_rew_lambda_' + str(arglist.DD_lambda) + '_beta_' + str(arglist.DD_beta) + '.npy')
        
        for _ in range(2):
            # ppo = Ppo_v2('BASE', base_env, arglist)
            # rewards.append(ppo.main_run_mpe())
            # noise_ppo = Ppo_v2('NOISE', noise_env, arglist)
            # trained_rewards.append(noise_ppo.main_run_mpe())
            ndd_ppo = NDD_Ppo_v2('NDD', noise_env_c, arglist)
            ndd_trained_rewards.append(ndd_ppo.main_run_mpe())
        
        # np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_org_rew.npy', np.array(rewards))
        # np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_noise_rew.npy', np.array(trained_rewards))
        np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_ndd_rew_lambda_' + str(arglist.DD_lambda) + '_beta_' + str(arglist.DD_beta) + '.npy', np.array(ndd_trained_rewards))
        
        # np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_ndd_rew_lambda_0_beta_0_risk' + str(arglist.DD_risk_sensitive) + str(arglist.DD_eta) + '_DMnum' + str(arglist.DM_num_epochs / arglist.DD_reward_sample_num) + '.npy', np.array(ndd_trained_rewards))

        # assert len(rewards) ==len(trained_rewards) == len(ndd_trained_rewards)                
        # visual_resault.line_graphs_01(np.arange(0, arglist.PPO_max_episodes_num, arglist.PPO_max_episodes_num / len(rewards[0])).data, {"ppo": rewards, "ppo noise": trained_rewards, "ppo ndd": ndd_trained_rewards}, ["iteration", "average reward"], 
        #                               os.getcwd() + '/PPO_v2' + str(arglist.res_dir) + str(arglist.env) + '/' + str(arglist.scene_name) + '/' + str(arglist.scene_name) + str(arglist.noisy_configuration[0]['distribution']['type']) + 
        #                               str(arglist.noisy_configuration[0]['distribution']['param']) + str(arglist.noisy_configuration[0]['reward_influence']) + 'ave_rew.pdf')
    
    elif arglist.env == "smac":
        base_env = make_env_SMAC(arglist)
        noise_env = make_env_SMAC_noise(arglist)
        noise_env_c = make_env_SMAC_noise(arglist)
        
        rewards = []
        trained_rewards = []
        ndd_trained_rewards = []

        win_rates = []
        trained_win_rates = []
        ndd_trained_win_rates = []
        
        for _ in range(2):
            ppo = Ppo_v1('BASE', base_env, arglist)
            _r, _w = ppo.main_run_smac()
            rewards.append(_r)
            win_rates.append(_w)
            
            noise_ppo = Ppo_v1('NOISE', noise_env, arglist)
            _r, _w = noise_ppo.main_run_smac()
            trained_rewards.append(_r)
            trained_win_rates.append(_w)
            
            ndd_ppo = NDD_Ppo_v1('NDD', noise_env_c, arglist)
            _r, _w = ndd_ppo.main_run_smac()
            ndd_trained_rewards.append(_r)
            ndd_trained_win_rates.append(_w)
        
        np.save(str(arglist.scene_name) + '_00_org_rew.npy', np.array(rewards))
        np.save(str(arglist.scene_name) + '_00_org_win.npy', np.array(win_rates))
        np.save(str(arglist.scene_name) + '_00_noise_rew.npy', np.array(trained_rewards))
        np.save(str(arglist.scene_name) + '_00_noise_win.npy', np.array(trained_win_rates))
        np.save(str(arglist.scene_name) + '_00_ndd_rew.npy', np.array(ndd_trained_rewards))
        np.save(str(arglist.scene_name) + '_00_ndd_win.npy', np.array(ndd_trained_win_rates))

def qmix_main_run(arglist):
    arglist.noisy_configuration = eval(arglist.noisy_configuration)
    arglist.scene_name = str(arglist.scene_name)
    if arglist.env == "mpe":
        base_env = make_env_MPE_v0(arglist, False)
        noise_env = make_env_MPE_v0_noise(arglist, True)
        # noise_env_c = make_env_MPE_noise(arglist)
        
        rewards = []
        trained_rewards = []
        ndd_trained_rewards = []
        
        for _ in range(2):
            # qmix = Qmix('BASE', base_env, arglist)
            # rewards.append(qmix.main_run_mpe())
            noise_qmix = Qmix('NOISE', noise_env, arglist)
            trained_rewards.append(noise_qmix.main_run_mpe())
            # ndd_qmix = NDD_Ppo_v2('NDD', noise_env_c, arglist)
            # ndd_trained_rewards.append(ndd_qmix.main_run_mpe())
        
        # np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_org_rew_qmix.npy', np.array(rewards))
        np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_noise_rew_' + str(arglist.algorithm) + '.npy', np.array(trained_rewards))
        # np.save(str(arglist.scene_name) + str(arglist.noisy_configuration[0]['distribution']['param']) + '_ndd_rew.npy', np.array(ndd_trained_rewards))

        pass
    elif arglist.env == "smac":
        base_env = make_env_SMAC(arglist)        
        rewards = []
        trained_rewards = []
        ndd_trained_rewards = []

        win_rates = []
        trained_win_rates = []
        ndd_trained_win_rates = []
    
        for _ in range(2):
            qmix = Qmix('BASE', base_env, arglist)
            _r, _w = qmix.main_run_smac()
            rewards.append(_r)
            win_rates.append(_w)

    pass

def ddpg_main_run(arglist):
    arglist.noisy_configuration = eval(arglist.noisy_configuration)
    arglist.scene_name = str(arglist.scene_name)
    if arglist.env == "mpe":
        base_env = make_env_MPE(arglist)
        noise_env = make_env_MPE_noise(arglist)
        
        rewards = []
        trained_rewards = []
        ndd_trained_rewards = []
        
        for _ in range(2):
            # ddpg = Ddpg_v2('BASE', base_env, arglist)
            # rewards.append(ddpg.main_run_mpe())
            noise_ddpg = Ddpg_v2('NOISE', noise_env, arglist)
            trained_rewards.append(noise_ddpg.main_run_mpe())
            # ndd_qmix = NDD_Ppo_v2('NDD', noise_env_c, arglist)
            # ndd_trained_rewards.append(ndd_qmix.main_run_mpe())
        
        # np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_org_rew_qmix.npy', np.array(rewards))
        np.save(str(arglist.scene_name) + str(len(arglist.noisy_configuration)) + '_noise_rew_' + str(arglist.algorithm) + '.npy', np.array(trained_rewards))
        # np.save(str(arglist.scene_name) + str(arglist.noisy_configuration[0]['distribution']['param']) + '_ndd_rew.npy', np.array(ndd_trained_rewards))
    elif arglist.env == "smac":
        # base_env = make_env_SMAC(arglist)
        noise_env = make_env_SMAC_noise(arglist)
                
        rewards = []
        trained_rewards = []
        ndd_trained_rewards = []

        win_rates = []
        trained_win_rates = []
        ndd_trained_win_rates = []
    
        for _ in range(2):
            noise_ddpg = Ddpg_v1('NOISE', noise_env, arglist)
            _r, _w = noise_ddpg.main_run_smac()
            trained_rewards.append(_r)
            trained_win_rates.append(_w)
        
        np.save(str(arglist.scene_name) + str(arglist.noisy_configuration[0]['distribution']['param']) + '_noise_rew_' + str(arglist.algorithm) + '.npy', np.array(trained_rewards))
        np.save(str(arglist.scene_name) + str(arglist.noisy_configuration[0]['distribution']['param']) + '_noise_win_' + str(arglist.algorithm) + '.npy', np.array(trained_win_rates))
            


if __name__ == '__main__':     
    arglist = arg_config.parse_args()
    print(arglist.algorithm)
    if arglist.algorithm == 'ppo':
        ppo_main_run(arglist)
    elif arglist.algorithm == 'qmix' or arglist.algorithm == 'vdn':
        qmix_main_run(arglist)
    elif arglist.algorithm == 'ddpg' or arglist.algorithm == 'td3':
        ddpg_main_run(arglist)