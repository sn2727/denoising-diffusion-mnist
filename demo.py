# Imports
import torch
import torchvision
import torchvision.datasets as datasets

#import datasets
import diffusers
import accelerate

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import PIL

import argparse

from config import config
from utils import * 
from model import *
from train import *



if __name__ == "__main__":
    
    # MNIST training data 
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    # Plots the first ten images
    plot_mnist_images(dataset)

    # Transform images 
    dataset = transform(dataset, config)

    # Plot first ten images with Gaussian noise added
    plot_mnist_images_noise(dataset)

    # loads model from checkpoint trained for 5 epochs 
    state_dict = torch.load('./model/checkpoint_5ep.pth')
    model.load_state_dict(state_dict)

    sample_image = sample(model, noise_scheduler, 2, "./figures")
    
    