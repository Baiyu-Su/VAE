import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAE import VAE
from utils import sample
from utils import loss_function
from utils import grid_generation
from utils import image_generation
import ssl
import requests
import time
import torchvision
import cv2

# initialize the training
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set hyperparameters
batch_size = 36
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
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# initialize the list to store loss vs epoch
num_of_epoch = np.arange(epoch) + 1
list_of_loss = np.zeros_like(num_of_epoch, dtype='float32')

# initialize the list to store original/reconstructed images
original_images_list = []
out_activated_images_list = []


# training the model
def train():
    for index in range(epoch):

        # set an adam optimizer with a decaying learning rate
        for data in train_loader:

            # load data
            img, _ = data
            img = img.to(device)

            optimizer.zero_grad()

            # retrieve the outputs from the neural network (out is the direct output without final activation)
            out, out_activated, mean, logVar = net(img)

            # calculate loss function
            loss = loss_function(mean, logVar, img, out, loss_type='BCE')

            # back prop
            loss.backward()
            optimizer.step()

            # store all the images produced in the last epoch
            if index == epoch - 1:
                original_images_list.append(img)
                out_activated_images_list.append(out_activated)

        # note the performance of each epoch
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch: {}, Loss: {}, time: {}'.format(index, loss, duration))
        list_of_loss[index] = loss.cpu().detach().item()

        # retrieve parameters for later use in CNN decoder
        CNN_parameters = net.CNN_parameters()

    # save parameters of this network after training
    torch.save(net.state_dict(), '/Users/byronsu/PycharmProjects/VAE_practice/nn_parameters/VAE_model.pt')

    # show the original images for training
    grid_generation(img_list=original_images_list,
                    save_path='/Users/byronsu/PycharmProjects/VAE_practice/images/original_images.png')

    # show the reconstructed images through VAE
    grid_generation(img_list=out_activated_images_list,
                    save_path='/Users/byronsu/PycharmProjects/VAE_practice/images/reconstruction_with_BCE.png')

    # show the sampled images
    sampled_images_list = [sample(CNN_parameters).cpu().detach() for i in range(10)]
    grid_generation(img_list=sampled_images_list,
                    save_path='/Users/byronsu/PycharmProjects/VAE_practice/images/sampling_with_BCE.png')

    # plot loss vs num of epoch
    plt.plot(num_of_epoch, list_of_loss)
    plt.savefig('/Users/byronsu/PycharmProjects/VAE_practice/images/loss_vs_epoch_with_BCE.png')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    path1 = '/Users/byronsu/PycharmProjects/VAE_practice/images/original_images.png'
    path2 = '/Users/byronsu/PycharmProjects/VAE_practice/images/reconstruction_with_BCE.png'
    save_path = '/Users/byronsu/PycharmProjects/VAE_practice/images/compare_BCE.png'
    train()
    image_generation(path1=path1, path2=path2, save_path=save_path)
