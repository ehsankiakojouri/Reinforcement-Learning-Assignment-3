import torch
import torch.nn as nn
import torch.optim as optim


# according to the world models paper appendix
Z_i = 1024
ACTION_DIM = 3
HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5
BATCH_SIZE = 32
EPOCHS = 20
# TEMPERATURE = 1.1

class RNN(nn.Module):
    """ MULTI STEPS forward.
        input: aggregated vector of latents(Z_i + actions)
        """
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=Z_i + ACTION_DIM, hidden_size=HIDDEN_UNITS)
        self.fc = nn.Linear(HIDDEN_UNITS, GAUSSIAN_MIXTURES * (3 * Z_i))

    def forward(self, x, h=None, c=None):
        if h is None:
            h = torch.zeros(1, HIDDEN_UNITS)
        if c is None:
            c = torch.zeros(1, HIDDEN_UNITS)
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

def MDN_RNN_loss(y_true, pi, mu, sigma):
    # Reshape mean and std tensors
    mu = mu.view(-1, GAUSSIAN_MIXTURES, Z_i)
    sigma = sigma.view(-1, GAUSSIAN_MIXTURES, Z_i)
    pi = pi.view(-1, GAUSSIAN_MIXTURES, Z_i)

    # Ensure y_true has the correct shape
    y_true = y_true.unsqueeze(1).expand_as(mu)
    # print(y_true.shape, mu.shape, sigma.shape, pi.shape)

    # Calculate the negative log likelihood loss for the MDN
    # print(mu, sigma)
    dist = torch.distributions.Normal(mu, sigma)
    weighted_probs = dist.log_prob(y_true) + pi.log()
    loss = -torch.logsumexp(weighted_probs, dim=1).mean()  # corrected line
    return loss
def main():
    start_batch = 0
    max_batch = 100
    new_model = True
    batch_size = 200
    rnn = RNN()
    optimizer = optim.RMSprop(rnn.parameters())

    # if not new_model:
    #     try:
    #         rnn.set_weights('./rnn/weights.h5')
    #     except:
    #         print("Either set --new_model or ensure ./rnn/weights.h5 exists")
    #         raise

    train_rnn_input = torch.load('./data/rnn_input.pth')
    train_rnn_output = torch.load('./data/rnn_output.pth')


    for epoch in range(EPOCHS):
        for i in range(0, 60000, batch_size):
            print(f'batch {i}')
            x = train_rnn_input[i:i+batch_size]
            y = train_rnn_output[i:i+batch_size]
            rnn.train()
            optimizer.zero_grad()
            rnn_output, _ = rnn(x)
            loss = MDN_RNN_loss(y, *rnn_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
            optimizer.step()
        # vae.decoder(rnn.)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}')

    # Save model weights
    torch.save(rnn.state_dict(), './models/rnn_weights.pth')

if __name__ == "__main__":
    main()
