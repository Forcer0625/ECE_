import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNNAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=64):
        super(RNNAgent, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
class MLPAgent(nn.Module):
    def __init__(self, observation_dim, n_actions, hidden_dim=128, device=torch.device('cpu')):
        super(MLPAgent, self).__init__()

        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, n_actions)
        self.device = device
        self.to(device)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.q(x)
    
class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim=64, device=torch.device('cpu')):
        super(QMixer, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.device = device

        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.n_agents * self.hidden_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim))

        self.hyper_b1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1))

        self.to(device)

    def forward(self, q_values, states):
        #states = states.reshape(-1, self.state_dim)
        #q_values = q_values.reshape(-1, 1, self.n_agents)
        
        w_1 = torch.abs(self.hyper_w1(states))
        w_1 = w_1.view(-1, self.n_agents, self.hidden_dim)
        b_1 = self.hyper_b1(states)
        b_1 = b_1.view(-1, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w_1) + b_1)

        w_2 = torch.abs(self.hyper_w2(states))
        w_2 = w_2.view(-1, self.hidden_dim, 1)
        b_2 = self.hyper_b2(states)
        b_2 = b_2.view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w_2 ) + b_2
        q_tot = q_tot.view(-1, 1)

        return q_tot
