from osim.env import L2M2019Env
import numpy as np
import torch

from SAC import SAC


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


