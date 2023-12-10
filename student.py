import cv2
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import CMAES
# student imports
import Generate_data, Generate_RNN_data, RNN, VAE, Controller
# parameters
v_path = 'models/VAE_weights.pth'
m_path = './models/rnn_weights.pth'
c_save_path = './models/cmaes_train_controller.pth'
vae_latent_size = 32
rnn_hidden_units = 256
# given by continuous environment is 3
action_dim = 3
gaussian_mixtures = 5
image_shape = (64, 64) # channels are received from gymnasium
class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make("CarRacing-v2")
        # VAE
        self.vae = VAE.VAE(vae_latent_size).to(device)
        self.vae.load_state_dict(torch.load(v_path))
        # Memory
        self.mdnrnn = RNN.RNN(vae_latent_size, rnn_hidden_units, action_dim, gaussian_mixtures).to(device)
        self.mdnrnn.load_state_dict(torch.load(m_path))
        # Controller
        self.control = Controller.CONTROLLER(rnn_hidden_units, vae_latent_size, action_dim).to(device)
        # self.control.load_state_dict(torch.load(m_path))
        # student initialization
        self.h, self.c = [torch.zeros(1, 1, rnn_hidden_units).to(self.device) for _ in range(2)]


    def forward(self, x):
        # TODO
        # I don't know what is forward supposed to do so I leave it as it is
        return x


    def pre_process(self, obs, target_image_shape):
        # resize and reshape for giving to the VAE
        x = cv2.resize(obs, dsize=(target_image_shape), interpolation=cv2.INTER_NEAREST)
        x = x.astype('float32') / 255
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).view(-1, obs.shape[-1], *target_image_shape)
        return x
    def act(self, state):
        # TODO
        # proper shape and size
        state = self.pre_process(state, image_shape).to(self.device)
        # obtaining representation of image
        z = self.vae.encoder(state)
        # using the policy to derive action from image representation and memory
        a = self.control.get_action(z, self.h)
        # since action is numpy array -> torch tensor
        a_tensor = torch.from_numpy(a).view(1, 1, action_dim).to(self.device)
        # aggregate action and image representation to update memory
        a_z_aggregated = torch.cat((z.view(1, 1, vae_latent_size), a_tensor), 2)
        _, (self.h, self.c) = self.mdnrnn.rnn(a_z_aggregated, (self.h, self.c))
        return a

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
