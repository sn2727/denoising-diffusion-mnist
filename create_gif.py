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
#sample(model, noise_scheduler, seed, "./figures/sample_images/")

#create_gif("./figures/reverse_diffusion.gif", duration=100)



def help():
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10, 5))
    axs = axs.ravel()
    images=["seed-1.png", "seed-2.png", "seed-3.png", "seed-4.png", "seed-5.png"]
    for i in range(5):
        image = Image.open("./figures/"+images[i]).resize((28,28))
        axs[i].imshow(image, cmap="grey")
        axs[i].axis('off')

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    plt.tight_layout()
    plt.savefig('./figures/generated_images.png', bbox_inches = 'tight', pad_inches = 0.1)

help()