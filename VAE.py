'''
a variational autoencoder
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

torch.manual_seed(0)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # 3 convolutional encoding layers
        self.encConv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding='same')
        self.encConv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5)
        self.encConv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)

        # two fully connected layers for mean & variance calculation
        self.encFC1 = nn.Linear(in_features=16 * 22 * 22, out_features=256)
        self.encFC2 = nn.Linear(in_features=16 * 22 * 22, out_features=256)

        # a fully connected layer and 3 deconvolution layers for decoding
        self.decFC = nn.Linear(in_features=256, out_features=16 * 22 * 22)
        self.decConv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3)
        self.decConv2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5)
        self.decConv3 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=5, padding=2)

    def encoder(self, tensor):
        # encoding -- implement the three convolutional layers
        tensor = F.relu(self.encConv1(tensor))
        tensor = F.relu(self.encConv2(tensor))
        tensor = F.relu(self.encConv3(tensor))
        self.cache_shape = tuple(tensor.shape)
        tensor = torch.flatten(tensor, start_dim=1)

        # derive the mean and variance
        mean = self.encFC1(tensor)
        logVar = self.encFC2(tensor)

        return mean, logVar

    # return a random variable with mean mean and variance Var
    def reparameterization(self, mean, logVar):
        # randx is a random value sampled from the standard normal distribution, std is standard deviation
        std = torch.exp_(0.5 * logVar)
        randx = torch.randn_like(std)

        return mean + std * randx

    def decoder(self, tensor):
        # decode data from the bottleneck layer to image
        tensor = F.relu(self.decFC(tensor))
        tensor = torch.reshape(tensor, self.cache_shape)
        tensor = F.relu(self.decConv1(tensor))
        tensor = F.relu(self.decConv2(tensor))

        # map the numbers back to range 0 - 1
        tensor = torch.sigmoid(self.decConv3(tensor))

        return tensor

    def forward(self, tensor):
        # execute feed forward neural network
        mean, logVar = self.encoder(tensor)
        tensor = self.reparameterization(mean, logVar)
        out = self.decoder(tensor)

        return out, mean, logVar
