import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using devide: {device}')

preprocess = transforms.Compose([
    transforms.ToTensor()
])

mnist_dataset_train = MNIST('./tmp/', train=True, download=True,
                            transform=preprocess)


class VAE(nn.Module):
    def __init__(self, nb_latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.z_mean_layer = nn.Linear(128, nb_latent)
        self.z_log_var_layer = nn.Linear(128, nb_latent)

        self.decoder = nn.Sequential(
            nn.Linear(nb_latent, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = x.view(x.size()[0], -1)
        x = self.encoder(x)
        return self.z_mean_layer(x), self.z_log_var_layer(x)

    def decode(self, x):
        x = self.decoder(x)
        return x.view(x.size()[0], 28, 28)

    @staticmethod
    def reparametrize(mu, logvar):
        eps = Variable(torch.FloatTensor(mu.size()).normal_())
        std = logvar.exp()
        return eps * std + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        x = self.reparametrize(mu, logvar)
        x = self.decode(x)
        return x, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    reconstruction_loss = F.binary_cross_entropy(recon_x.view(-1, 28 * 28),
                                                 x.view(-1, 28 * 28),
                                                 size_average=False)

    kl_element = 1 + logvar - mu ** 2 - logvar.exp() ** 2
    kl_loss = - beta * 0.5 * torch.sum(kl_element)
    return reconstruction_loss + kl_loss


def train(nb_latent=2, beta=1):
    model = VAE(nb_latent).to(device)
    data_loader = DataLoader(mnist_dataset_train, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    for epoch in range(30):
        avg_loss = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)

            outputs, z_mean, z_log_var = model(inputs)
            loss = loss_function(outputs, inputs, z_mean, z_log_var, beta)
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch} loss: {avg_loss/len(mnist_dataset_train)}')
    return model


def collect_latents(model):
    data_loader = DataLoader(mnist_dataset_train, batch_size=16, shuffle=False)
    latents_mean = []
    latents_log_var = []
    all_targets = []
    for inputs, targets in data_loader:
        inputs = Variable(inputs)
        all_targets.append(targets)

        latent_mean, latent_log_var = model.encode(inputs)
        latents_mean.append(latent_mean.data)
        latents_log_var.append(latent_log_var.data)
    latents_mean = torch.cat(latents_mean, dim=0).numpy()
    latents_log_var = torch.cat(latents_log_var, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    return pd.DataFrame({'x': latents_mean[:, 0],
                         'y': latents_mean[:, 1],
                         'vx': latents_log_var[:, 0],
                         'vy': latents_log_var[:, 1],
                         't': all_targets})


def plot_digits(model):
    n = 15
    figure = np.zeros((28 * n, 28 * n))

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            sample = np.array([[xi, yi]])
            sample = Variable(torch.from_numpy(sample).float())
            digit = model.decode(sample).data.numpy()[0]
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x
