from dataclasses import dataclass

## Parameters for training 

@dataclass
class Config:
    image_size = 32
    train_batch_size = 32
    eval_batch_size = 32
    num_epochs = 5
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmpup_steps = 500
    mixed_precision = 'fp16'
    seed = 0
    
config = Config()