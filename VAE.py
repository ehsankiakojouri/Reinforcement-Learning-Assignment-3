import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable

Z_DIM = 32
EPOCHS = 40
BATCH_SIZE = 32

class Sampling(nn.Module):
    def forward(self, args):
        z_mean, z_log_var = args
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(z_log_var / 2) * epsilon

class VAE(nn.Module):
    def __init__(self):
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
        )

        self.z_mean = nn.Linear(2*2*256, Z_DIM)
        self.z_log_var = nn.Linear(2*2*256, Z_DIM)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM, 1024),
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
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling()([z_mean, z_log_var])
        # Decode
        x_decoded = self.decoder(z)
        return x_decoded, z, z_mean, z_log_var


# Loss functions
def vae_r_loss(y_true, y_pred):
    return 10 * torch.mean((y_true.view(-1) - y_pred.view(-1)) ** 2)

def vae_kl_loss(z_mean, z_log_var):
    return -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))

def vae_loss(y_true, y_pred, z_mean, z_log_var):
    return vae_r_loss(y_true, y_pred) + vae_kl_loss(z_mean, z_log_var)

# Training loop
def train(vae, optimizer, data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2):
    data_size = len(data)
    split_index = int(data_size * (1 - validation_split))
    for epoch in range(epochs):
        vae.train()
        for i in range(0, split_index, batch_size):
            batch_data = data[i:i+batch_size]
            batch_data = Variable(batch_data)

            optimizer.zero_grad()
            recon_batch, z, z_mean, z_log_var = vae(batch_data)
            loss = vae_loss(batch_data, recon_batch, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
        # Validation loss
        vae.eval()
        val_data = Variable(data[split_index:])
        val_recon, val_z, val_z_mean, val_z_log_var = vae(val_data)
        val_loss = vae_loss(val_data, val_recon, val_z_mean, val_z_log_var)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        if epoch+1%5 == 0:
            ax1 = plt.subplot(2, 2, 1)
            ax1.imshow(val_recon[-1].detach().numpy().T)
            ax2 = plt.subplot(2, 2, 2)
            ax2.imshow(val_data[-1].detach().numpy().T)
            plt.show()
            torch.save(vae.state_dict(), f'./models/obs_data_car_racing_{epoch+1}.VAE_model')
    return vae

def main():
    data = torch.load('./data/obs_data_car_racing.pth')
    plt.imshow(data[50].detach().numpy().T)
    plt.title(data.shape)
    plt.show()
    vae = VAE()
    optimizer = optim.Adam(vae.parameters())
    vae = train(vae, optimizer, data)
    torch.save(vae.state_dict(), './models/obs_data_car_racing.VAE_model')

if __name__ == "__main__":
    main()
