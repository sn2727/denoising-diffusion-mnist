import matplotlib.pyplot as plt 
import numpy as np
import torchvision
import torch

from model import *

from config import config 

def plot_mnist_images(dataset):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
    axs = axs.ravel()
    
    for i in range(10):
        image = np.array(dataset[i][0]).reshape(28, 28)
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./figures/original_images.png')


def plot_mnist_images_noise(dataset): 
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
    axs = axs.ravel()
    
    for i in range(10):
        noise = torch.randn(dataset[0][0].shape)
        timesteps = torch.LongTensor([199])
        noisy_image = noise_scheduler.add_noise(dataset[0][0], noise, timesteps)
        image = torchvision.transforms.ToPILImage()(noisy_image.squeeze(1)).resize((256,256))
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./figures/noisy_images.png')
    plt.show()


def transform(dataset, config):
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (config.image_size, config.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: 2*(x-0.5)),
        ]
    )
    data = [(preprocess(image), label) for image, label in dataset]
    return data