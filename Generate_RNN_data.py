import torch
from VAE import VAE

def main():
    batch_size = 256
    vae = VAE()
    vae.load_state_dict(torch.load('models/obs_data_car_racing.VAE_model'), strict= False)
    obs_data = torch.load('./data/obs_data_car_racing.pth')
    action_data = torch.load('./data/action_data_car_racing.pth')

    print('Generating RNN data...')

    NO_samples = obs_data.shape[0]
    latent_shape = vae.encoder(obs_data[0:1]).shape
    action_shape = action_data[0].shape
    rnn_input = torch.zeros((NO_samples, latent_shape[-1] + action_shape[0]))
    rnn_output = torch.zeros((NO_samples, latent_shape[-1]))


    for i in range(0, obs_data.shape[0], batch_size):
        print('batch:', i)
        batch_action_data = action_data[i:i+batch_size]
        batch_obs_data = obs_data[i:i+batch_size]
        rnn_z_input = vae.encoder(batch_obs_data)
        aggregate_z_action = torch.cat([rnn_z_input, batch_action_data], dim=1) # aggregate z and action
        rnn_input[i : i+batch_size] = aggregate_z_action # input = z_i + action
        rnn_output[i : i+batch_size] = rnn_z_input

    # z_i -> z_i+1 since output of the model is => z_i+1
    shifted_rnn_output = torch.roll(rnn_output, shifts=1, dims=1)

    # drop the first and the last element of the
    # dataset since it is inconsistent with RNN(z_i+1 | z_i, a, h)
    trimmed_rnn_output = shifted_rnn_output[1:-1]
    trimmed_rnn_input = rnn_input[1:-1]
    print(f"RNN input shape: {trimmed_rnn_input.shape}")
    print(f"RNN output shape: {trimmed_rnn_output.shape}")
    print("Saving RNN dataset ...")
    torch.save(trimmed_rnn_input, './data/rnn_input.pth')
    torch.save(trimmed_rnn_output, './data/rnn_output.pth')

if __name__ == "__main__":
    main()
