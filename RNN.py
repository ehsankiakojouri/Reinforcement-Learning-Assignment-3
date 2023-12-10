import torch
import torch.nn as nn
import torch.optim as optim



class RNN(nn.Module):
    """ MULTI STEPS forward.
        input: aggregated vector of latents(Z_i + actions)
        """
    def __init__(self, vae_latent_size, hidden_units, action_dim, gaussian_mixtures):
        super(RNN, self).__init__()
        self.hidden_units = hidden_units
        self.rnn = nn.LSTM(input_size=vae_latent_size + action_dim, hidden_size=hidden_units)
        self.fc = nn.Linear(hidden_units, gaussian_mixtures * (3 * vae_latent_size))

    def forward(self, x, h=None, c=None):
        if h is None:
            h = torch.zeros(1, self.hidden_units)
        if c is None:
            c = torch.zeros(1, self.hidden_units)
        lstm_out, (h, c) = self.rnn(x, (h, c))
        pi, mu, sigma = self.mixture_density_network(lstm_out)
        return (pi, mu, sigma), (h, c)

    def mixture_density_network(self, y_pred):
        # fc_out.shape = (batch_size, number, 3) where number is hidden state size which is
        # irrelevant and 3 is for mean, standard deviation and probability tensors respectively
        fc_out = self.fc(y_pred).reshape(y_pred.size(0), -1, 3)
        _pi = fc_out[:, :, 0]
        mu = fc_out[:, :, 1]
        _sigma = fc_out[:, :, 2]
        # Normalize to turn it to a valid probability distribution (non-negative, sum to 1, etc.)
        pi = torch.softmax(_pi, dim=1)
        # for numerical stability & ensuring sigma > 0
        sigma = torch.exp(_sigma)
        # mu can be anything
        return pi, mu, sigma

def MDN_RNN_loss(y_true, pi, mu, sigma, gaussian_mixtures, latent_size):
    # Reshape mean and std tensors
    mu = mu.view(-1, gaussian_mixtures, latent_size)
    sigma = sigma.view(-1, gaussian_mixtures, latent_size)
    pi = pi.view(-1, gaussian_mixtures, latent_size)

    # Ensure y_true has the correct shape
    y_true = y_true.unsqueeze(1).expand_as(mu)
    # print(y_true.shape, mu.shape, sigma.shape, pi.shape)

    # Calculate the negative log likelihood loss for the MDN
    # print(mu, sigma)
    dist = torch.distributions.Normal(mu, sigma)
    weighted_probs = dist.log_prob(y_true) + pi.log()
    loss = -torch.logsumexp(weighted_probs, dim=1).mean()  # corrected line
    return loss


# according to the world models paper appendix
def main(latent_size=32, action_dim=3, hidden_units=256, gaussian_mixtures=5, batch_size=500, epochs=20, rnn_weights_path='./models/rnn_weights.pth', rnn_input_data_path='./data/rnn_input.pth', rnn_output_data_path='./data/rnn_output.pth'):
    rnn = RNN(latent_size, hidden_units, action_dim, gaussian_mixtures)
    optimizer = optim.RMSprop(rnn.parameters())

    train_rnn_input = torch.load(rnn_input_data_path)
    train_rnn_output = torch.load(rnn_output_data_path)


    for epoch in range(epochs):
        for i in range(0, len(train_rnn_output), batch_size):
            x = train_rnn_input[i:i+batch_size]
            y = train_rnn_output[i:i+batch_size]
            rnn.train()
            optimizer.zero_grad()
            rnn_output, _ = rnn(x)
            loss = MDN_RNN_loss(y, *rnn_output, gaussian_mixtures, latent_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    # Save model weights
    torch.save(rnn.state_dict(), rnn_weights_path)

if __name__ == "__main__":
    main()
