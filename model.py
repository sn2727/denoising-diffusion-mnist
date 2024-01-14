import diffusers
from config import config 

import torch
import os 
import torchvision

model = diffusers.UNet2DModel(
    sample_size=config.image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128,128,256,512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=200)

@torch.no_grad()
def sample(unet, scheduler, seed, save_process_dir=None):
    torch.manual_seed(seed)
    
    if save_process_dir:
        if not os.path.exists(save_process_dir):
            os.mkdir(save_process_dir)
    
    # How many timestpes for backward process
    timesteps = 200
    scheduler.set_timesteps(timesteps)
    image = torch.randn((1,1,32,32)).to(model.device)
    num_steps = max(noise_scheduler.timesteps).numpy()
    
    for t in noise_scheduler.timesteps:
        model_output = unet(image,t)['sample']
        image = scheduler.step(model_output, int(t), image, generator=None)['prev_sample']
        if save_process_dir:
            save_image = torchvision.transforms.ToPILImage()(image.squeeze(0))
            save_image.resize((256,256)).save(
                os.path.join(save_process_dir,"seed-"+str(seed)+"_"+f"{num_steps-t.numpy():03d}"+".png"),format="png")
        
    return torchvision.transforms.ToPILImage()(image.squeeze(0))