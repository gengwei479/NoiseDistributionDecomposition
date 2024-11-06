import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser("param configuration")
    
    # training parameters ppo,vdn,qmix,ddpg,td3
    parser.add_argument("--algorithm", type = str, default = 'ppo', help = "base algorithm")
    
    # PPO for decentralized scenes
    parser.add_argument("--PPO_v0_max_train_steps", type=int, default=int(1e5), help=" Maximum number of training steps")#1e5
    parser.add_argument("--PPO_v0_evaluate_freq", type=float, default=100, help="Evaluate the policy every 'evaluate_freq' steps")#5000
    parser.add_argument("--PPO_v0_evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--PPO_v0_save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--PPO_v0_batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--PPO_v0_mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--PPO_v0_rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--PPO_v0_mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--PPO_v0_lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--PPO_v0_gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--PPO_v0_lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--PPO_v0_epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--PPO_v0_K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--PPO_v0_use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--PPO_v0_use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--PPO_v0_use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--PPO_v0_entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--PPO_v0_use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--PPO_v0_use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--PPO_v0_use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--PPO_v0_set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--PPO_v0_use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--PPO_v0_use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--PPO_v0_add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--PPO_v0_use_agent_specific", type=float, default=True, help="Whether to use agent specific global state.")
    parser.add_argument("--PPO_v0_use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument("--PPO_v0_use_global_state", type=bool, default=False, help="Whether to use global info")
    
    # PPO for continuous scenes
    parser.add_argument("--PPO_max_episodes_num", type = int, default = 3000, help = "number of episodes")
    parser.add_argument("--PPO_save_rate", type=int, default = 20, help="save model once every time this many episodes are completed")
    parser.add_argument("--PPO_buffer_size", type = int, default = 2048, help = "buffer max size")
    parser.add_argument("--PPO_batch_size", type = int, default = 64, help = "mini-batch size")
    parser.add_argument("--PPO_train_times", type = int, default = 10, help = "training times for every epoch")
    parser.add_argument("--PPO_actor_learning_rate", type = float, default = 0.0003, help = "learning rate of A or C networks")
    parser.add_argument("--PPO_critic_learning_rate", type = float, default = 0.0003, help = "learning rate of A or C networks")
    parser.add_argument("--PPO_layer_dim", type = int, default = 256, help = "units number of A or C networks")
    parser.add_argument("--PPO_policy_clip", type = int, default = 0.2, help = "clip range of policy gradient")
    parser.add_argument("--PPO_lambda", type = int, default = 0.75, help = "GAE parameter")#0.95 0.75 
    parser.add_argument("--PPO_gamma", type = int, default = 0.99, help = "discount factor")
    parser.add_argument("--PPO_entropy_coef", type=float, default=0.01, help="policy entropy")
    
    # QMIX and VDN
    parser.add_argument("--QMIX_max_train_steps", type=int, default=int(3000), help=" Maximum number of training steps")#1e5
    parser.add_argument("--QMIX_episode_limit", type=int, default=int(100), help=" Maximum episode of training steps")#
    parser.add_argument("--QMIX_evaluate_freq", type=float, default=100, help="Evaluate the policy every 'evaluate_freq' steps")#5000
    parser.add_argument("--QMIX_evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--QMIX_save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--QMIX_epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--QMIX_epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--QMIX_epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--QMIX_buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--QMIX_batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--QMIX_lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--QMIX_gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--QMIX_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--QMIX_hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--QMIX_hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--QMIX_rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--QMIX_mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--QMIX_use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--QMIX_use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--QMIX_use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--QMIX_use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--QMIX_use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--QMIX_add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--QMIX_add_agent_id", type=bool, default=False, help="Whether to add agent id into the observation")
    parser.add_argument("--QMIX_use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--QMIX_use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--QMIX_use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--QMIX_target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--QMIX_tau", type=int, default=0.005, help="If use soft update")
    
    # QMIX for mpe
    parser.add_argument('--QMIX_v1_gamma', type=float, default=0.99, help="Discount factor for env")
    parser.add_argument('--QMIX_v1_episode_length', type=int, default=25, help="Max length for any episode")
    parser.add_argument('--QMIX_v1_buffer_size', type=int, default=5000, help="Max # of transitions that replay buffer can contain")
    parser.add_argument('--QMIX_v1_hypernet_layers', type=int, default=2, help="Number of layers for hypernetworks. Must be either 1 or 2")
    parser.add_argument('--QMIX_v1_mixer_hidden_dim', type=int, default=32, help="Dimension of hidden layer of mixing network")
    parser.add_argument('--QMIX_v1_hypernet_hidden_dim', type=int, default=64, help="Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2")
    parser.add_argument('--QMIX_v1_share_policy', action='store_false', default=False, help="Whether use a centralized critic") 
    parser.add_argument('--QMIX_v1_use_feature_normlization', action='store_true', default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument('--QMIX_v1_use_orthogonal', action='store_false', default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument('--QMIX_v1_use_ReLU', action='store_false', default=True, help="Whether to use ReLU")
    parser.add_argument('--QMIX_v1_layer_N', type=int, default=1, help="Number of layers for actor/critic networks")
    parser.add_argument('--QMIX_v1_hidden_size', type=int, default=64, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument('--QMIX_v1_lr', type=float, default=0.0005, help="Learning rate for RMSProp")
    parser.add_argument('--QMIX_v1_batch_size', type=int, default=32, help="Number of episodes to train on at once")
    parser.add_argument("--QMIX_v1_opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--QMIX_v1_weight_decay", type=float, default=0)
    parser.add_argument("--QMIX_v1_gain", type=float, default=1)
    parser.add_argument('--QMIX_v1_prev_act_inp', action='store_true', default=False, help="Whether the actor input takes in previous actions as part of its input")
    parser.add_argument('--QMIX_v1_chunk_len', type=int, default=80, help="Time length of chunks used to train via BPTT")
    parser.add_argument('--QMIX_v1_grad_norm_clip', type=float, default=10.0, help="Max gradient norm (clipped if above this value)")
    parser.add_argument('--QMIX_v1_double_q', type=bool, default=True, help="Whether to use double q learning")
    parser.add_argument('--QMIX_v1_num_env_steps', type=int, default=3000, help="Number of env steps to train for")
    parser.add_argument('--QMIX_v1_num_random_episodes', type=int, default=5, help="Number of episodes to add to buffer with purely random actions")
    parser.add_argument('--QMIX_v1_epsilon_start', type=float, default=1.0, help="Starting value for epsilon, for eps-greedy exploration")
    parser.add_argument('--QMIX_v1_epsilon_finish', type=float, default=0.05, help="Ending value for epsilon, for eps-greedy exploration")
    parser.add_argument('--QMIX_v1_epsilon_anneal_time', type=int, default=5000, help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument('--QMIX_v1_use_soft_update', action='store_true', default=False, help="Whether to use soft update")
    parser.add_argument('--QMIX_v1_tau', type=float, default=0.01, help="Polyak update rate")
    parser.add_argument('--QMIX_v1_hard_update_interval_episode', type=int, default=200, help="After how many episodes the lagging target should be updated")
    parser.add_argument('--QMIX_v1_train_interval_episode', type=int, default=1, help="Number of episodes between updates to actor/critic")
    parser.add_argument('--QMIX_v1_train_interval', type=int, default=1, help="Number of episodes between updates to actor/critic")
    parser.add_argument('--QMIX_v1_test_interval', type=int,  default=100, help="After how many episodes the policy should be tested")
    parser.add_argument('--QMIX_v1_save_interval', type=int, default=50000, help="After how many episodes of training the policy model should be saved")
    parser.add_argument('--QMIX_v1_log_interval', type=int, default=10000, help="After how many episodes of training the policy model should be saved")
    parser.add_argument('--QMIX_v1_num_test_episodes', type=int, default=32, help="How many episodes to collect for each test")
    
    #DDPG for smac
    parser.add_argument("--DDPG_v1_max_train_steps", type=int, default=20000, help=" Maximum number of training steps")#1e5
    parser.add_argument("--DDPG_v1_evaluate_freq", type=float, default=100, help="Evaluate the policy every 'evaluate_freq' steps")#5000
    parser.add_argument("--DDPG_v1_evaluate_times", type=float, default=32, help="Evaluate times")
    #DDPG for mpe
    parser.add_argument("--DDPG_v2_max_episodes_num", type = int, default = 3000, help = "number of episodes")
    parser.add_argument("--DDPG_v2_buffer_size", type=int, default=1024, help="The capacity of the replay buffer")#1024
    parser.add_argument("--DDPG_v2_batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--DDPG_v2_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--DDPG_v2_noise_std_init", type=float, default=0.01, help="The std of Gaussian noise for exploration")
    parser.add_argument("--DDPG_v2_noise_std_min", type=float, default=0.0001, help="The std of Gaussian noise for exploration")
    parser.add_argument("--DDPG_v2_noise_decay_steps", type=float, default=3e5, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--DDPG_v2_use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--DDPG_v2_lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--DDPG_v2_lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--DDPG_v2_gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--DDPG_v2_tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--DDPG_v2_use_orthogonal_init", type=bool, default=False, help="Orthogonal initialization")
    parser.add_argument("--DDPG_v2_use_grad_clip", type=bool, default=False, help="Gradient clip")
    parser.add_argument("--DDPG_v2_policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--DDPG_v2_noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--DDPG_v2_policy_update_freq", type=int, default=10, help="The frequency of policy updates")
    
    # environment
    parser.add_argument("--env", type = str, default = "smac", help = "which enviornment to use")#mpe smac
    parser.add_argument("--env_continuous", type = bool, default = True, help = "RL enviornment is continuous or not")
    
    # mpe
    ## simple_adversary, simple_crypto, simple_speaker_listener, simple_spread, simple_tag
    
    # smac
    # '3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '27m_vs_30m', 'MMM', 'MMM2', '2s3z', '3s5z', '3s5z_vs_3s6z',
    ## '3s_vs_3z', '3s_vs_4z', '3s_vs_5z', '1c3s5z', '2m_vs_1z', 'corridor', '6h_vs_8z', '2s_vs_1sc', 'so_many_baneling', 'bane_vs_bane', '2c_vs_64zg'
    parser.add_argument("--scene_name", type = str, default = "3m", help = "name of the RL scene")#
    parser.add_argument("--cooperative_agents", type = list, default = [], help = "which agents will cooperate")
    parser.add_argument("--dim", type = int, default = 2, help = "dimension of the RL scene")
    parser.add_argument("--agent_num", type = list, default = [], help = "the number list of different agents")#[1, 1, 1]
    parser.add_argument("--state_dim", type = list, default = [], help = "state dim of all agents")#[8, 10, 10]
    parser.add_argument("--scene_range", type = list, default = [], help = "scene space of all agents")#[8, 10, 10]
    parser.add_argument("--action_dim", type = list, default = [], help = "action dim of all agents")#[5, 5, 5]
    parser.add_argument("--action_range", type = dict, default = {}, help = "action space range")#{'shape': [1], 'range': [0, 1]}
    parser.add_argument("--reward_range", type = dict, default = [], help = "reward space range")#[min, max]  
    
    # personal settings
    parser.add_argument("--model_dir", type = str, default = "/save/", help = "directory in which training state and model should be saved")
    parser.add_argument("--log_dir", type = str, default = "/log/", help = "directory in which log files should be saved")
    parser.add_argument("--res_dir", type = str, default = "/res/", help = "directory in which pictrues or result figures should be saved")
    
    # noise configurations
    '''
    distribution format:
    {
        "type": "guass" #Continuous distribution
        "param": {"mu": *, "sigma": *}
    } or
    {
        "type": "uniform"
        "param": {"low": *, "high": *}
    } or
    {
        "type": "gamma"
        "param": {"alpha": *, "beta": *}
    } or
    {
        "type": "beta"
        "param": {"alpha": *, "beta": *}
    } or
    {
        "type": "exponential"
        "param": {"lambd": *}
    }
    '''
    parser.add_argument("--noisy_configuration", type = json.loads, default = 
                        [
                            # {"size": 0.2, 
                            #  "pos": [],# len(pos) = 0 means random generation
                            #  "category": "state_related_only", 
                            #  "reward_influence": "additive_noise",
                            #  "distribution": {
                            #     "type": "gauss", 
                            #     "param":{"mu": 0, "sigma": 1}
                            #     }
                            #  },
                            {"size": 5, 
                            "pos": [0, 0],
                            "reward_influence": "additive_noise",
                            "reward_influence_weight": 1,
                            "distribution": {
                                "type": "gauss", 
                                "param":{"mu": 0, "sigma": 15}
                                }
                            }                          
                        ], help = "noise settings")
    
    # noise-decompose training
    parser.add_argument("--DD_agent_network_layer_dim", type = int, default = 64, help = "units number of agent dist-networks")
    parser.add_argument("--DD_lambda", type = float, default = 1, help = "super param lambda")
    parser.add_argument("--DD_beta", type = float, default = 1, help = "super param beta")
    parser.add_argument("--DD_network_learning_rate", type = int, default = 0.003, help = "units number of dist-networks")#0.003
    parser.add_argument("--DD_reward_partition_length", type = int, default = 7, help = "partition length for evenly spaced reward range")
    parser.add_argument("--DD_loss_fun", type = str, default = "mse", help = "loss function, norm regularize or not")
    parser.add_argument("--DD_reward_sample_num", type = int, default = 500, help = "sample times for reward noise")#500
    parser.add_argument("--DD_episode_num", type = int, default = 50, help = "number of DD_episodes")#50
    parser.add_argument("--DD_epsilon", type = float, default = 0.001, help = "dist PDF loss threshold")
    parser.add_argument("--DD_use_expection_norm", type = bool, default = True)
    parser.add_argument("--DD_expection_norm", type = float, default = 10)
    parser.add_argument("--DD_use_dist_norm", type = bool, default = True)
    parser.add_argument("--DD_dist_norm", type = float, default = 1)
    parser.add_argument("--DD_use_weight_norm", type = bool, default = True)
    parser.add_argument("--DD_weight_norm", type = float, default = 1)
    parser.add_argument("--DD_risk_sensitive", type = str, default = "none", help = "risk tendency about policy to choose")#CPW 0.71 WANG 0.75 -0.75 POW -2 CVaR 0.25 0.1
    parser.add_argument("--DD_eta", type = float, default = 0, help = "parameters of risk tendency")
    
    # diffusion model
    parser.add_argument("--DM_enable", type = bool, default = False)#False True
    parser.add_argument("--DM_agent_network_layer_dim", type = int, default = 128, help = "units number of dm-networks")
    parser.add_argument("--DM_diffusion_depth", type = int, default = 3, help = "diffusion times of dm-networks")
    parser.add_argument("--DM_learning_rate", type = float, default = 0.001, help = "learning rate of dm-networks")
    parser.add_argument("--DM_num_epochs", type = int, default = 100, help = "update times of dm-networks")#100
    parser.add_argument("--DM_input_dim", type = int, default = 50, help = "input dim of dm-networks")#50
    
    return parser.parse_args()
