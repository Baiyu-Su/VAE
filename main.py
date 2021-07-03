import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAE import VAE
import ssl
import requests
import time
import torchvision

# initialize the training
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set hyperparameters
batch_size = 72
epoch = 30
learning_rate = 1e-3

# request the MNIST dataset from the website
requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Datasets
train_dataset = datasets.MNIST('~/PycharmProjects/VAE_practice',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST('~/PycharmProjects/VAE_practice',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# Data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1
)

# set the neural network and Adam optimizer
net = VAE().to(device)


# training the model
def train():
    for index in range(epoch):

        # set an adam optimizer with a decaying learning rate
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate / (1 + index))
        for data in train_loader:
            img, _ = data
            img = img.to(device)

            # retrieve the outputs from the neural network
            out, mean, logVar = net(img)

            # calculate loss function
            KL_divergence = 0.5 * torch.sum(-logVar + mean ** 2 + torch.exp_(logVar) - 1)
            L2_loss = ((out - img) ** 2).sum()
            loss = KL_divergence + L2_loss

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # select a neat result from the epochs
        if index == 0:
            cache_loss = loss

        if loss < 0.85 * cache_loss and index > 10:
            break

        # note the performance of each epoch
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch: {}, Loss: {}, KL: {}, L2: {}'.format(index, loss, KL_divergence, L2_loss))
        print(duration)

    # show the reproducted figures
    examples = out.clone().detach()

    '''for i in range(16):
        example = torch.reshape(examples[i, :, :], (28, 28))
        plt.imshow(example, interpolation='nearest')
        plt.title('Epoch ' + str(index))
        plt.show()'''
    # grid of numbers
    grid = 1 - torchvision.utils.make_grid(examples, nrow=6)
    channels, width, height = list(grid.shape)
    grid = torch.reshape(grid, (width, height, channels))
    plt.imshow(grid)
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    train()
