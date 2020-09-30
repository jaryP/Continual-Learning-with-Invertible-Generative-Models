import numpy as np
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_size, z_dim, n_hidden, conditioning_size=0):
        super(VAE, self).__init__()

        lrs = np.linspace(input_size, z_dim, n_hidden, dtype=int)
        print(lrs)

        encoder, decoder = [], []

        for i in range(1, len(lrs)-1):
            encoder.extend([nn.Linear(lrs[i-1], lrs[i]), nn.ReLU()])
        lrs[-1] += conditioning_size
        for i in range(1, len(lrs)):
            decoder.extend([nn.Linear(lrs[-i], lrs[-i-1]), nn.ReLU()])
        decoder.pop()

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    #     self.fc1 = nn.Linear(784, 400)
        self.mu = nn.Linear(lrs[-2], z_dim)
        self.sigma = nn.Linear(lrs[-2], z_dim)
    #     self.fc3 = nn.Linear(20, 400)
    #     self.fc4 = nn.Linear(400, 784)
    #
    # def encode(self, x):
    #     h1 = F.relu(self.fc1(x))
    #     return self.fc21(h1), self.fc22(h1)
    #

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std
    #
    # def decode(self, z):
    #     h3 = F.relu(self.fc3(z))
    #     return torch.sigmoid(self.fc4(h3))
    #

    def forward(self, x, y=None):
        # mu, logvar = self.encode(x.view(-1, 784))
        enc = self.encoder(x)

        mu = self.mu(enc)
        logvar = self.sigma(enc)

        z = mu + torch.exp(0.5*logvar)*torch.randn_like(logvar)
        if y is not None:
            z = torch.cat([z, y], 1)

        return self.decoder(z), mu, logvar

    def backward(self, u, y=None):
        if y is not None:
            u = torch.cat([u, y], 1)
        return self.decoder(u), None

# lrs = np.linspace(100, 10, 5, dtype=int)[1:-2]

# vae = VAE(100, 5, 25, conditioning_size=100)
# print(vae)