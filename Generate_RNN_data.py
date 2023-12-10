import torch
from VAE import VAE

def main(vae_latent_shape=32, batch_size=500, vae_weights_path='models/VAE_weights.pth', action_data_path='./data/action_data_car_racing.pth', obs_data_path='./data/obs_data_car_racing.pth', rnn_input_data_path='./data/rnn_input.pth', rnn_output_data_path='./data/rnn_output.pth'):

    vae = VAE(vae_latent_shape)
    vae.load_state_dict(torch.load(vae_weights_path), strict= False)
    obs_data = torch.load(obs_data_path)
    action_data = torch.load(action_data_path)

    print('Generating RNN data...')

    NO_samples = obs_data.shape[0]
    action_shape = action_data[0].shape
    rnn_input = torch.zeros((NO_samples, vae_latent_shape + action_shape[0]))
    rnn_output = torch.zeros((NO_samples, vae_latent_shape))


    for i in range(0, obs_data.shape[0], batch_size):
        print('batch:', i)
        batch_action_data = action_data[i:i+batch_size]
        batch_obs_data = obs_data[i:i+batch_size]
        _, rnn_z_input, _, _ = vae(batch_obs_data)
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
    torch.save(trimmed_rnn_input, rnn_input_data_path)
    torch.save(trimmed_rnn_output, rnn_output_data_path)

if __name__ == "__main__":
    main()
