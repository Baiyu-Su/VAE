import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np


# apply sampling from multivariate Gaussian and pass through the decoder
def sample(CNN_parameters):
    std = torch.exp_(0.5 * CNN_parameters['logVar'])
    randx = torch.randn_like(std)
    rand_var = CNN_parameters['mean'] + std * randx
    tensor = F.relu(CNN_parameters['decFC'](rand_var))
    tensor = torch.reshape(tensor, CNN_parameters['cache_shape'])
    tensor = F.relu(CNN_parameters['decConv1'](tensor))
    tensor = torch.sigmoid(CNN_parameters['decConv2'](tensor))

    return tensor


# loss function with BCE/MSE loss + KL divergence loss
def loss_function(mean, logVar, img, out, loss_type='BCE'):

    if loss_type == 'BCE':
        # thresholding the data (img) to 1 or 0 (black or white)
        img = torch.where(img > 0.5, 1.0, 0.0)
        KL_divergence = 0.5 * torch.sum(-logVar + mean.pow(2) + torch.exp_(logVar) - 1)
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, img, reduction='sum')
        loss = KL_divergence + cross_entropy_loss
        return loss

    if loss_type == 'MSE':
        KL_divergence = 0.5 * torch.sum(-logVar + mean.pow(2) + torch.exp_(logVar) - 1)
        mse_loss = F.mse_loss(out, img, reduction='sum')
        loss = KL_divergence + mse_loss
        return loss

    else:
        raise NameError('unwanted loss type')


# make a grid plot with 100 images
def grid_generation(img_list, save_path):
    images_tensor = torch.cat(img_list, dim=0)
    examples = images_tensor.clone().detach()
    # select first 100 numbers to generate a grid
    output_examples = examples[0:100, :, :, :]
    grid = torchvision.utils.make_grid(output_examples, nrow=10, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()


# concatenate two images together to compare
def image_generation(path1, path2, save_path):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    comb_img = cv2.hconcat([img1, img2])
    cv2.imwrite(save_path, comb_img)

