from osim.env import L2M2019Env
import numpy as np
import torch

import wrapper
from SAC import SAC
import replay_buffer
from replay_buffer import Data_Processor
from replay_buffer import Transition

#-----------------------------------------------------------------
# arguments

'''
agent = Agent(input_channels=skip, 
                n_actions=n_actions, 
                input_dim=len_of_obs_vec,
                reward_scale=reward_scale, 
                entropy_coe=entropy_coe,
                soft_update_coe=soft_update_tau, 
                batch_size=batch_size,
                gamma=discounted_factor,
                device=device,
                q1_dict=q1_dict,
                q2_dict=q2_dict,
                policy_dict=policy_dict)
'''

class Agent:
    def __init__(self) -> None:

        skip = 4
        n_actions = 22
        len_of_obs_vec = 339
        reward_scale = 20.
        discounted_factor = .99
        entropy_coe = 1. / reward_scale
        soft_update_tau = 0.005
        batch_size = 256
        discounted_factor = .99
        device = torch.device("cpu")

        self.input_channels = skip
        self.n_actions = n_actions
        self.gamma = discounted_factor
        self.early_exploration = 0
        self.batch_size = batch_size
        self.input_dim = len_of_obs_vec

        self.current_obs = None
        self.action = None
        self.counter = 0
        self.skip = skip

        self.device = device
        self.sac = SAC(n_actions=n_actions, 
                       reward_scale=reward_scale, 
                       entropy_coe=entropy_coe, 
                       dicounted_factor=discounted_factor,
                       input_dim=self.input_dim, 
                       soft_update_coe=soft_update_tau,
                       device=device,
                       q_learning_rate=1e-4,
                       policy_learning_rate=3e-5)
        
        nets_history = torch.load('110062371_hw4_data', map_location=torch.device('cpu'))
        q1_dict=nets_history['q1']
        q2_dict=nets_history['q2']
        policy_dict=nets_history['policy']

        self.sac.load_nets(dict_q1= q1_dict, dict_q2=q2_dict, dict_policy=policy_dict)

    def unpack_dict_obs(self, observation:dict):
        res = []
        if not isinstance(observation, dict):
            if not (isinstance(observation, np.ndarray) or isinstance(observation, list)):
                res.append(observation)
            else:
                for element in observation:
                    res = res + self.unpack_dict_obs(element)
            return res
        
        for key in observation:
            res = res + self.unpack_dict_obs(observation[key])

        return res

    @torch.no_grad()
    def act(self, observation):

        #if self.counter == 0:
        observation =self.unpack_dict_obs(observation)
        if self.current_obs is None:
            self.current_obs = observation

        if np.abs((np.array(self.current_obs[:22]) - np.array(observation[:22]))).mean() >= .1:
            self.counter = 0
            #print(np.abs((np.array(self.current_obs[:22]) - np.array(observation[:22]))).mean())

        self.current_obs = observation 

        observation = torch.tensor(observation, device=self.device, dtype=torch.float).unsqueeze(0)

        if self.counter % 4 == 0:
            self.action, _ = self.sac.get_action_and_log_prob(observation=observation)
            self.action = self.action.squeeze().detach().cpu().__array__()
            #print(self.counter)
        self.counter += 1

        #action = action.squeeze().detach().cpu().__array__()
        scaled_action = self.action.copy()
        scaled_action = (scaled_action + 1.) / 2.
        #scaled_action = np.clip(scaled_action, 0., 1.)

        #print(scaled_action[:5])

        return scaled_action
#-----------------------------------------------------------------

#env = L2M2019Env(difficulty=2, visualize=False)
#env = wrapper.Wrapper_For_SkipFrame(env, 4)
#env = wrapper.Vector_Observation(env)
#obs = env.reset()

#-----------------------------------------------------------------
# training

# agent = Agent()


# obs = env.reset()

# ep_len = 0

# for j in range(5, 6):

#     nets_history = torch.load('110062371_hw4_data copy {}'.format(j), map_location=torch.device('cpu'))
#     #nets_history = torch.load('110062371_hw4_data', map_location=torch.device('cpu'))
#     print('110062371_hw4_data copy {} loaded.'.format(j))
#     q1_dict=nets_history['q1']
#     q2_dict=nets_history['q2']
#     policy_dict=nets_history['policy']

#     agent.sac.load_nets(dict_q1= q1_dict, dict_q2=q2_dict, dict_policy=policy_dict)

#     cumulative_reward = 0.
#     cumulative_reward_2 = 0.

#     for i in range(20):
#         ep_len = 0
#         obs = env.reset()
#         while True:
#             action = agent.act(obs)

#             next_obs, reward, done, info = env.step(action)

#             if ep_len < 1000:
#                 cumulative_reward += reward
#             cumulative_reward_2 += reward
#             ep_len += 1

#             obs = next_obs

#             if done or ep_len >= 1000:
#                 break
#         print('reward: {}, eplen: {}'.format(cumulative_reward / (i + 1), ep_len))
#     print(cumulative_reward / 20.)

