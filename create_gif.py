from model import *

from PIL import Image
import os

import matplotlib.pyplot as plt 
import numpy as np


def create_gif(output_path, duration=100):
    images = []
    files = sorted(os.listdir("./figures/sample_images"))[150:200]

    for file in files:
        if file.endswith(".png"):
            images.append(Image.open(f"./figures/sample_images/{file}"))

    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

seed = 2
state_dict = torch.load('./model/checkpoint_5ep.pth')
model.load_state_dict(state_dict)

sample(model, noise_scheduler, seed, "./figures/sample_images/")

create_gif("./figures/reverse_diffusion.gif", duration=100)