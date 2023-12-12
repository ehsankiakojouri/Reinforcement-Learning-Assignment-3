import random
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean).to(device)
        return z_mean + torch.exp(z_log_var / 2) * epsilon


class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.z_mean = nn.Linear(2 * 2 * 256, latent_size).to(device)
        self.z_log_var = nn.Linear(2 * 2 * 256, latent_size).to(device)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 1, 1)),
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling()(z_mean, z_log_var)
        # Decode
        x_decoded = self.decoder(z)
        return x_decoded, z, z_mean, z_log_var


# Loss functions
def vae_r_loss(y_true, y_pred):  # regularization
    return 10 * torch.mean((y_true.view(-1) - y_pred.view(-1)) ** 2).to(device)


def vae_kl_loss(z_mean, z_log_var):  # K-L loss
    return -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var)).to(device)


def vae_loss(y_true, y_pred, z_mean, z_log_var):
    return vae_r_loss(y_true, y_pred) + vae_kl_loss(z_mean, z_log_var)


# Training loop
def train(vae, optimizer, data, epochs, batch_size, validation_split=0.2):
    data_size = len(data)
    split_index = int(data_size * (1 - validation_split))
    for epoch in range(1, epochs + 1):
        vae.train()
        for i in range(0, split_index, batch_size):
            print(f'batch size {batch_size} starting at: {i}')
            batch_data = data[i:i + batch_size].to(device)
            batch_data = Variable(batch_data)

            optimizer.zero_grad()
            recon_batch, z, z_mean, z_log_var = vae(batch_data)
            loss = vae_loss(batch_data, recon_batch, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
        # Validation loss
        vae.eval()
        val_data = Variable(data[split_index:].to(device))
        val_recon, val_z, val_z_mean, val_z_log_var = vae(val_data)
        val_loss = vae_loss(val_data, val_recon, val_z_mean, val_z_log_var)

        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        if epoch % 5 == 0:
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(val_recon[-1].detach().numpy().transpose((1, 2, 0)))
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(val_data[-1].detach().numpy().transpose((1, 2, 0)))
            plt.show()
    return vae


def main(latent_size, epochs, batch_size, validation_split, data_path, vae_weights_path):
    data = torch.load(data_path).to(device)
    random_index = random.randint(0, len(data) - 1)
    plt.imshow(data[random_index].detach().numpy().transpose((1, 2, 0)))
    plt.title(f"random image: #{random_index}")
    plt.xlabel(f"dataset shape: {data.shape}")
    plt.savefig('myplot')
    plt.show()
    vae = VAE(latent_size).to(device)
    optimizer = optim.Adam(vae.parameters())
    vae = train(vae, optimizer, data, epochs, batch_size, validation_split)
    torch.save(vae.state_dict(), vae_weights_path)

# parameters
v_path = 'models/VAE_weights.pth'
obs_data_path = './data/obs_data_car_racing.pth'
vae_latent_size = 32
vae_batch_size = 32
vae_epochs = 10


if __name__ == "__main__":
    main(latent_size=vae_latent_size, epochs=vae_epochs, batch_size=vae_batch_size, validation_split=0.2,
         # validation_split is for printing loss for VAE during training
         data_path=obs_data_path,
         vae_weights_path=v_path)
