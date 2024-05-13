import torch
import torch.nn as nn


class Embedding_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(Embedding_Net, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU(.2)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, observation):
        x = nn.Flatten()(observation)

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)

        return x
    
class RND:
    def __init__(self, input_size, device, hidden_size=128, output_size=64, learning_rate=3e-4) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.beta = 20.

        self.device = device

        self.predictor = Embedding_Net(self.input_size, self.hidden_size, self.output_size).to(device)
        self.embedding = Embedding_Net(self.input_size, self.hidden_size, self.output_size).to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    def get_instrinsic_reward_and_update(self, observation):
        predicted_embedding = self.predictor(observation)
        embedding = self.embedding(observation)

        reward = torch.mean(((predicted_embedding - embedding.detach()) ** 2.), dim=1)

        loss = reward.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward.detach().unsqueeze(-1) * self.beta
