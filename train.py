import torch
import torchvision
import torchvision.datasets as datasets

import accelerate
import diffusers

import argparse

from tqdm.auto import tqdm

from config import *
from model import * 
from utils import * 

def train(config, model, optimizer, lr_scheduler, train_loader, noise_scheduler):

    accelerator = accelerate.Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_loader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_loader):
            clean_images, labels = batch
            #clean_images = batch['images']

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            batch_size = clean_images.shape[0]

            # Sample a set of random time steps for each image in mini-batch
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), device=clean_images.device)
            
            noisy_images=noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps)["sample"]
                loss = torch.nn.functional.mse_loss(noise_pred,noise)
                accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {
                "loss" : loss.detach().item(),
                "lr" : lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
    
    accelerator.unwrap_model(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=float, help="How many epochs to train", default=5)
    parser.add_argument("-l", "--learning_rate", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("-b", "--batch_size", type=float, help="Batch size", default=32)

    args = parser.parse_args()

    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.train_batch_size = args.batch_size
    config.eval_batch_size = args.batch_size

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    dataset = transform(dataset, config)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                                          num_warmup_steps=config.lr_warmpup_steps, 
                                                                          num_training_steps=len(train_loader) * config.num_epochs)

    
    train(config, model, optimizer, lr_scheduler, train_loader, noise_scheduler) 
    torch.save(model.state_dict(), './model/checkpoint.pth')