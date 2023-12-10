import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import CMAES
# student imports
import Generate_data, Generate_RNN_data, RNN, VAE, Controller
# My system has 6 CPU cores
class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make("CarRacing-v1")

    def forward(self, x):
        # TODO
        print('move! move! move!')
        return x
    
    def act(self, state):
        # TODO
        # state is an observation(image)
        # feed it to VAE encoder
        # feed the results to RNN
        # aggregate the VAE representation of image with RNN hidden state
        # feed the aggregated matrix to controller
        print('act! act! act!')
        return 

    def train(self):
        # TODO
        Generate_data.main()
        VAE.main()
        Generate_RNN_data.main()
        RNN.main()
        CMAES.main()
        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
