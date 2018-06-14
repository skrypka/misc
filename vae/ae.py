import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using devide: {device}')

preprocess = transforms.Compose([
    transforms.ToTensor()
])

mnist_dataset_train = MNIST('./tmp/', train=True, download=True,
                            transform=preprocess)


class Autoencoder(nn.Module):
    def __init__(self, nb_latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nb_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nb_latent, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        x = self.decoder(x)
        return x.view(x.size()[0], 28, 28)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


def train(nb_latent=2):
    model = Autoencoder(nb_latent).to(device)
    criterion = nn.BCELoss(size_average=False).to(device)
    data_loader = DataLoader(mnist_dataset_train, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    for epoch in range(20):
        avg_loss = 0
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 28 * 28), inputs.view(-1, 28 * 28))
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch} loss: {avg_loss/len(mnist_dataset_train)}')
    return model


def collect_latents(model):
    data_loader = DataLoader(mnist_dataset_train, batch_size=16, shuffle=False)
    latents = []
    all_targets = []
    for inputs, targets in data_loader:
        inputs = Variable(inputs)
        all_targets.append(targets)

        latent = model.encode(inputs).data
        latents.append(latent)
    latents = torch.cat(latents, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    return pd.DataFrame({'x': latents[:, 0],
                         'y': latents[:, 1],
                         't': all_targets})


def plot_digits(model):
    n = 15
    figure = np.zeros((28 * n, 28 * n))

    grid_x = np.linspace(-25, 25, n)
    grid_y = np.linspace(-25, 25, n)

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


if __name__ == "__main__":
    model = train(nb_latent=2)
