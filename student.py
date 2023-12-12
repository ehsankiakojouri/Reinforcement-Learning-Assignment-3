import cv2
import gymnasium as gym
import torch
import torch.nn as nn

import CMAES
# student imports
import VAE_Generate_data, Generate_RNN_data, RNN, VAE, Controller

# parameters
v_path = 'models/VAE_weights.pth'
m_path = './models/rnn_weights.pth'
c_path = './models/cmaes_train_controller.pth'
action_data_path = './data/action_data_car_racing.pth'
obs_data_path = './data/obs_data_car_racing.pth'
rnn_output_data_path = './data/rnn_output.pth'
rnn_input_data_path = './data/rnn_input.pth'
vae_latent_size = 32
rnn_hidden_units = 256
# given by continuous environment is 3
action_dim = 3
render_during_VAE_data_gen = False
gaussian_mixtures = 5
image_shape = (64, 64)  # channels are received from gymnasium
vae_batch_size = 32
vae_epochs = 10
RNN_epochs = 20
RNN_train_batch_size = 256
RNN_data_gen_batch_size = 256
target_reward = 930
initial_time_steps = 50
CMAES_population_size = 64
CMAES_initial_sigma = 0.1
# (best in name is because it is obtained from the best individual in CMAES algorithm)
NO_best_rollouts = 100  # average number of rollouts for agent to have the target reward
eval_best_interval = 25  # the interval that best solution of CMAES is evaluated
# (best in name is because it is calculated for all individuals regardless of their cost in CMAES algorithm)
normal_rollouts = 16  # average number of rollouts for agent to have the target reward
VAE_data_gen_episodes = 200
VAE_data_gen_time_steps_each_episode = 300


class Policy(nn.Module):
    continuous = True  # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make("CarRacing-v2")
        # VAE
        self.vae = VAE.VAE(vae_latent_size).to(device)
        # Memory
        self.mdnrnn = RNN.RNN(vae_latent_size, rnn_hidden_units, action_dim, gaussian_mixtures).to(device)
        # Controller
        self.control = Controller.CONTROLLER(rnn_hidden_units, vae_latent_size, action_dim).to(device)
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
        _, z, _, _ = self.vae(state)
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
        VAE_Generate_data.main(total_episodes=VAE_data_gen_episodes, time_steps=VAE_data_gen_time_steps_each_episode,
                               render=render_during_VAE_data_gen, image_size=(3, *image_shape),
                               # 3 is number of channels
                               action_data_path=action_data_path,
                               obs_data_path=obs_data_path)
        VAE.main(latent_size=vae_latent_size, epochs=vae_epochs, batch_size=vae_batch_size, validation_split=0.2,
                 # validation_split is for printing loss for VAE during training
                 data_path=obs_data_path,
                 vae_weights_path=v_path)
        Generate_RNN_data.main(vae_latent_shape=vae_latent_size, batch_size=RNN_data_gen_batch_size,
                               vae_weights_path=v_path,
                               action_data_path=action_data_path, obs_data_path=obs_data_path,
                               rnn_input_data_path=rnn_input_data_path, rnn_output_data_path=rnn_output_data_path)
        RNN.main(latent_size=vae_latent_size, action_dim=action_dim, hidden_units=rnn_hidden_units,
                 gaussian_mixtures=gaussian_mixtures,
                 batch_size=RNN_train_batch_size, epochs=RNN_epochs, rnn_weights_path=m_path,
                 rnn_input_data_path=rnn_input_data_path, rnn_output_data_path=rnn_output_data_path)
        CMAES.main(initial_time_steps=initial_time_steps, vae_latent_size=vae_latent_size,
                   best_rollouts=NO_best_rollouts, n_rollouts=normal_rollouts, pop_size=CMAES_population_size,
                   eval_best_interval=eval_best_interval, target=target_reward, sigma=CMAES_initial_sigma,
                   rnn_hidden_units=rnn_hidden_units,
                   action_dim=action_dim, gaussian_mixtures=gaussian_mixtures, v_path=v_path, m_path=m_path,
                   c_path=c_path)
        return

    def save(self):
        print('since all the savings are done during training in each corresponding step, save function is obsolete')

    def load(self):
        # VAE
        self.vae.load_state_dict(torch.load(v_path))
        # Memory
        self.mdnrnn.load_state_dict(torch.load(m_path))
        # Controller
        controller_data = torch.load(c_path)
        self.reward = controller_data['reward']  # maximum reward obtained by CMAES
        self.generation = controller_data['generation']  # at which generation it was obtained
        state_dict = controller_data['state_dict']  # controller model itself
        self.control.load_state_dict(state_dict)
        print(self.reward, self.generation)
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
