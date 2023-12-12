import torch
from torch import nn


class CONTROLLER(nn.Module):
    # World models c-model. A fully connected layer
    def __init__(self, rnn_hidden_units, vae_latent_size, action_dim):
        super().__init__()
        self.rnn_hidden_units = rnn_hidden_units
        self.vae_latent_size = vae_latent_size
        self.action_dim = action_dim
        self.fc1 = nn.Linear(rnn_hidden_units+vae_latent_size, action_dim, bias=True)
    def get_action(self, z, h):
        i = torch.cat((z, h.view(1, self.rnn_hidden_units)), 1)
        y = self.fc1(i).view(3)

        y = y.tanh()
        y[1] = (y[1] + 1) / 2
        y[2] = (y[2] + 1) / 2
        return y.detach().cpu().numpy()
