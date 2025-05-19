import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
# Do not modify the input of the 'act' function and the '__init__' function. 
class Pi_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size, 384),
            nn.ReLU(),
            nn.Linear(384, 512),
            nn.ReLU(),
        )
        self.mu = torch.nn.Linear(512, action_size)
        self.log_sigma = torch.nn.Linear(512, action_size)

    def forward(self, x, not_random=False, use_logprob=False):
        y2 = self.layers(x)
        mu = self.mu(y2)
        if not_random:
            log_prob = None
            action = torch.tanh(mu)
        else:
            log_sigma = torch.clamp(self.log_sigma(y2),min=-20.0,max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_sample = dist.rsample()
            if use_logprob:
                log_prob = dist.log_prob(x_sample).sum(1)
                log_prob = log_prob - (2*np.log(2) - 2*x_sample - 2*F.softplus(-2*x_sample)).sum(1)
            else:
                log_prob = None
            action = torch.tanh(x_sample)

        return action, log_prob
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.actor = Pi_FC()
        self.checkpoint = torch.load('actor.ckpt') 
        self.actor.load_state_dict(self.checkpoint['actor'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor(state, True, False)           
        return action
