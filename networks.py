import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, num_actions, input_dim) -> None:
        super(Q_Network, self).__init__()

        self.linear1 = nn.Linear(input_dim + num_actions, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.norm2 = nn.LayerNorm(1024)
        self.linear3 = nn.Linear(1024, 1)

    def forward(self, observation, action):
        x = nn.Flatten()(observation)
        x = torch.cat([x, action], 1)

        x = self.linear1(x)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.linear3(x)

        return x

class Policy_Net(nn.Module):
    def __init__(self, num_actions, input_dim) -> None:
        super(Policy_Net, self).__init__()

        self.linear1 = nn.Linear(input_dim, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.norm2 = nn.LayerNorm(1024)

        self.mu = nn.Linear(1024, num_actions)
        self.log_std = nn.Linear(1024, num_actions)

    def forward(self, observation):

        x = nn.Flatten()(observation)
        x = F.elu(self.linear1(x))
        x = self.norm1(x)
        x = F.elu(self.linear2(x))
        x = self.norm2(x)

        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std