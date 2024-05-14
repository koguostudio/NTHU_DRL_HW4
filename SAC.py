import torch
import torch.nn as nn
from torch.distributions import Normal
from networks import Q_Network, Policy_Net
from RND import RND

class SAC:
    def __init__(self, n_actions, reward_scale, entropy_coe, dicounted_factor, soft_update_coe, input_dim, device:torch.device, q_learning_rate=3e-4, policy_learning_rate=3e-4, curiosity=False, n_step=1) -> None:
        self.n_actions = n_actions
        self.device = device
        self.q_learning_rate = q_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.reward_scale = reward_scale
        self.gamma = dicounted_factor
        self.entropy_coe = entropy_coe
        self.cumulative_entropy_coe = entropy_coe * ((1. - self.gamma ** n_step)/(1. - self.gamma))
        self.tau = soft_update_coe
        self.input_dim = input_dim
        self.curiosity = curiosity
        self.n_step = n_step

        # self.target_entropy = -self.n_actions
        # self.log_entorpy_alpha = torch.tensor(0, dtype=torch.float32, requires_grad=True, device=self.device)
        # self.entropy_alpha = self.log_entorpy_alpha.exp().item()

        self.q_net = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net2 = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net_target = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net2_target = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net2_target.load_state_dict(self.q_net2.state_dict())

        self.policy_net = Policy_Net(num_actions=self.n_actions, input_dim=input_dim).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.q_learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q_net2.parameters(), lr=self.q_learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_learning_rate)
        #self.alpha_optimizer = torch.optim.Adam([self.log_entorpy_alpha], lr=1e-3)
        self.loss_function = nn.MSELoss()
        self.loss_function2 = nn.MSELoss()

        if self.curiosity:
            self.rnd = RND(input_size=input_dim, device=device)

    def load_nets(self, dict_q1, dict_q2, dict_policy):
        self.q_net.load_state_dict(dict_q1)
        self.q_net2.load_state_dict(dict_q2)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net2_target.load_state_dict(self.q_net2.state_dict())
        self.policy_net.load_state_dict(dict_policy)

    def save_nets(self):
        torch.save(dict(q1=self.q_net.state_dict(), q2=self.q_net2.state_dict(), policy=self.policy_net.state_dict()), '110062371_hw4_data')

    def get_action_and_log_prob(self, observation):
        '''
        observation-> (batch_size, ...)
        action0 -> -1~1
        action1 -> 0~1
        action2 -> 0~1
        '''
        mean, log_std = self.policy_net(observation)
        std = log_std.exp()

        z = Normal(0, 1).sample(mean.shape).to(self.device)
        #print(z.shape, std.shape)

        unscaled_action = mean + std*z
        action = torch.tanh(unscaled_action)

        '''
        1st action : a = tanh(u) ; da/du = 1 - tanh(u) ^2
        2~3 actions: a = (tanh(u) + 1) / 2 ; da/du = (1 - tanh(u) ^2) / 2 => log(da/du) = log(1 - tanh(u) ^2) - log(2)

        we obtain the log prob. of a by applying Jocobian
        ''' 

        log_prob = Normal(mean, std).log_prob(unscaled_action) - torch.log(1 - action ** 2 + 1e-6)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob
    
    def get_action(self, observation):
        mean, log_std = self.policy_net(observation)

        action = torch.tanh(mean)

        return action

    def update_nets(self, action, current_observation, next_observation, reward, done):
        #-----------------------------------------------------------------------------------
        # update Q nets
        
        q1 = self.q_net(current_observation, action)
        q2 = self.q_net2(current_observation, action)
        
        predicted_action, log_prob = self.get_action_and_log_prob(current_observation)
        predicted_next_action, next_log_prob = self.get_action_and_log_prob(next_observation)
        q1_target = self.q_net_target(next_observation, predicted_next_action)
        q2_target = self.q_net2_target(next_observation, predicted_next_action)

        # reward scaling
        #reward = self.reward_scale * (reward - reward.mean().item()) / (reward.std().item() + 1e-6)

        if self.curiosity:
            intrinsic_reward = self.rnd.get_instrinsic_reward_and_update(next_observation)
            total_reward = reward + intrinsic_reward
        else:
            total_reward = reward

        y = total_reward + (1. - done.float()) * (self.gamma ** self.n_step) * (torch.min(q1_target, q2_target) - self.cumulative_entropy_coe * next_log_prob)
        
        loss1 = self.loss_function(q1, y.detach())
        self.q_optimizer.zero_grad()
        loss1.backward()
        self.q_optimizer.step()

        loss2 = self.loss_function2(q2, y.detach())
        self.q2_optimizer.zero_grad()
        loss2.backward()
        self.q2_optimizer.step()

        #-----------------------------------------------------------------------------------
        # update policy
        q1_policy = self.q_net(current_observation, predicted_action)
        q2_policy = self.q_net2(current_observation, predicted_action)
        selected_q = torch.min(q1_policy, q2_policy)

        policy_loss = (self.entropy_coe * log_prob - selected_q).mean()
        #print(self.entropy_coe * log_prob, selected_q)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        #alpha_loss = -(self.log_entorpy_alpha * (log_prob + self.target_entropy).detach())

        # self.alpha_optimizer.zero_grad()
        # alpha_loss = alpha_loss.mean()
        # alpha_loss.backward()
        # self.alpha_optimizer.step()
        #self.entropy_alpha = self.log_entorpy_alpha.exp().item()
        #-----------------------------------------------------------------------------------
        # soft update
        q1_dict = self.q_net.state_dict()
        q2_dict = self.q_net2.state_dict()
        q1_target_dict = self.q_net_target.state_dict()
        q2_target_dict = self.q_net2_target.state_dict()
        for key in q1_dict:
            q1_target_dict[key] = q1_target_dict[key] * (1. - self.tau) + q1_dict[key] * self.tau


        for key in q2_dict:
            q2_target_dict[key] = q2_target_dict[key] * (1. - self.tau) + q2_dict[key] * self.tau

        self.q_net_target.load_state_dict(q1_target_dict)
        self.q_net2_target.load_state_dict(q2_target_dict)

        #print(y, reward)
        if self.curiosity:
            return y, torch.min(q1, q2), loss1.item(), loss2.item(), policy_loss.item(), reward.mean().item(), intrinsic_reward.mean().item()
        else:
            return y, torch.min(q1, q2), loss1.item(), loss2.item(), policy_loss.item()
    
    @torch.no_grad()
    def get_q_values(self, action, current_observation, next_observation, reward, done):
        q1 = self.q_net(current_observation, action)
        q2 = self.q_net2(current_observation, action)
        q = torch.min(q1, q2)

        predicted_next_action, next_log_prob = self.get_action_and_log_prob(next_observation)
        q1_target = self.q_net_target(next_observation, predicted_next_action)
        q2_target = self.q_net2_target(next_observation, predicted_next_action)

        y = reward + (1. - done.float()) * (self.gamma ** self.n_step) * (torch.min(q1_target, q2_target) - self.cumulative_entropy_coe * next_log_prob)
        return y, q