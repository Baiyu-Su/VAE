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


# initialize the training
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# set hyperparameters
batch_size = 64
epoch = 8
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
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# training the model
def train():
    for index in range(epoch):

        for data in train_loader:
            img, _ = data
            img = img.to(device)

            # retrieve the outputs from the neural network
            out, mean, logVar = net(img)

            # calculate loss function
            KL_divergence = 0.5 * torch.sum(-logVar + mean ** 2 + torch.exp_(logVar) - 1)
            cross_entropy = F.binary_cross_entropy(out, img)
            loss = KL_divergence + cross_entropy

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ''' calculate the training duration for the sake that 
        MacOS cannot be configured with cuda and it's extremely slow 
        to train locally on CPU'''
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch: {}, Loss: {}'.format(index, loss))
        print(duration)


if __name__ == "__main__":
    start_time = time.time()
    train()
