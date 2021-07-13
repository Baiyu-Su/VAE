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
batch_size = 32
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


# set the neural network and Adam optimizer
net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# initialize the list to store loss vs epoch
train_loss = 0
num_of_epoch = np.arange(epoch) + 1
list_of_training_loss = np.zeros_like(num_of_epoch, dtype='float32')
list_of_validation_loss = np.zeros_like(num_of_epoch, dtype='float32')

# initialize the list to store original/reconstructed images
original_images_list = []
out_activated_images_list = []


# training the model
def train(loss_type):
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

    for index in range(epoch):

        training_loss = 0
        validation_loss = 0

        # set an adam optimizer with a decaying learning rate
        for data in train_loader:

            # load data
            img, _ = data
            img = img.to(device)

            optimizer.zero_grad()

            # retrieve the outputs from the neural network (out is the direct output without final activation)
            out, out_activated, mean, logVar = net(img)

            # calculate loss function and the average loss for each image over the whole dataset
            loss = loss_function(mean, logVar, img, out, out_activated, loss_type=loss_type)
            training_loss += loss.item()/len(train_loader.sampler)

            # back prop
            loss.backward()
            optimizer.step()

        # apply VAE on testing dataset
        for data in test_loader:

            img, _ = data
            img = img.to(device)

            out, out_activated, mean, logVar = net(img)

            loss = loss_function(mean, logVar, img, out, out_activated, loss_type=loss_type)
            validation_loss += loss.item()/len(test_loader.sampler)

        # note the performance of each epoch
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch: {}, Loss: {}, Time: {}'.format(index, training_loss, duration))
        list_of_training_loss[index] = training_loss
        list_of_validation_loss[index] = validation_loss

    # save parameters of this network after training
    torch.save(net.state_dict(),
               '~/PycharmProjects/VAE_practice/nn_parameters/VAE_model_'+str(loss_type)+'.pt')

    # plot loss vs num of epoch
    plt.plot(num_of_epoch, list_of_training_loss, '-r', label='avg train loss')
    plt.plot(num_of_epoch, list_of_validation_loss, '-b', label='avg valid loss')
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('loss')
    plt.savefig('~/PycharmProjects/VAE_practice/images/loss_vs_epoch_with_'+str(loss_type)+'.png')
    plt.show()


# evaluate the performance of VAE on the train dataset after training
def evaluate(loss_type):

    # reconfigure the train loader to save running time
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=100,
    )

    # retrieve the first batch of images from the dataset
    data = next(iter(train_loader))

    # load data
    img, _ = data
    img = img.to(device)
    original_images_list.append(img)

    # pass the data trough network to find the reconstructed images
    _, out_activated, _, _ = net(img)
    out_activated_images_list.append(out_activated)

    # retrieve parameters for later use in CNN decoder
    CNN_parameters = net.CNN_parameters()

    # show the original images for training
    grid_generation(img_list=original_images_list,
                    save_path='~/PycharmProjects/VAE_practice/images/original_images_'
                              +str(loss_type)+'.png')

    # show the reconstructed images through VAE
    grid_generation(img_list=out_activated_images_list,
                    save_path='~/PycharmProjects/VAE_practice/images/reconstruction_with_'
                              +str(loss_type)+'.png')

    # show the sampled images
    sampled_images_list = [sample(CNN_parameters).cpu().detach()]
    grid_generation(img_list=sampled_images_list,
                    save_path='~/PycharmProjects/VAE_practice/images/sampling_with_'
                              +str(loss_type)+'.png')


if __name__ == "__main__":
    start_time = time.time()
    path1 = '~/PycharmProjects/VAE_practice/images/original_images_BCE.png'
    path2 = '~/PycharmProjects/VAE_practice/images/reconstruction_with_BCE.png'
    save_path = '~/PycharmProjects/VAE_practice/images/compare_BCE.png'
    train(loss_type='BCE')
    evaluate(loss_type='BCE')
    image_generation(path1=path1, path2=path2, save_path=save_path)
