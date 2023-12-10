import torch
from torch import nn


class CONTROLLER(nn.Module):
    ''' World models c-model. A fully connected layer '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256+1024, 3, bias=True)
    ''' Feed forward z-vector and hidden state. Scale returned action to valid action space.

    Parameters
    ----------
    z : torch.Tensor, shape=(1, 1, 32)
        The z-vector latent space represenation of the current observation

    h : torch.Tensor, shape=(1, 1, 256)
        The hidden state of the LSTM in the m-model at the current time step
    '''
    def get_action(self, z, h):
        i = torch.cat((z, h.view(1, 256)), 1)
        y = self.fc1(i).view(3)

        y = y.tanh()
        y[1] = (y[1] + 1) / 2
        y[2] = (y[2] + 1) / 2
        return y.detach().cpu().numpy()
