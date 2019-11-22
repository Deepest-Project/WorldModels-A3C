import torch
import torch.nn as nn
import numpy as np
from hparams import HyperParams as hp

class VAE(nn.Module):
    def __init__(self, latent_dims, img_channels=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_dims)
        self.decoder = Decoder(img_channels, latent_dims)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # sigma = logsigma.exp()
        z = self.reparam(mu, logvar)
        y = self.decoder(z)
        return y, mu, logvar

    def reparam(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        z = eps*std + mu
        # z = eps*sigma + mu
        return z

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super(Encoder, self).__init__()
        # flatten_dims = hp.img_height//2**4
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 32, 48, 48)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 64, 24, 24)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 128, 12, 12)
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 256, 6, 6)
        )
        self.fc = nn.Linear(6*6*256, latent_dims*2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # (B, d)
        h = self.fc(h) # (B, )
        mu = h[:, :self.latent_dims]
        logvar = h[:, self.latent_dims:]
        # sigma = self.softplus(h[:, self.latent_dims:])
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dims, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 6, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 128, 6, 6)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 64, 12, 12)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 32, 24, 24)
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(), # (B, 32, 48, 48)
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            # nn.Tanh()
            nn.Sigmoid()
            # nn.LeakyReLU(), # (B, c, 96, 96)
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, 2, 2)
        y = self.decoder(h)
        return y


def vae_loss(recon_x, x, mu, logvar):
    """ VAE loss function """
    recon_loss = nn.MSELoss(size_average=False)
    BCE = recon_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD