import torch 
import numpy as np

def get_mean_std(Dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in Dataloader:
        channels_sum += torch.mean(data, dim= [0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim = [0, 2, 3])
        num_batches += 1

    mean = channels_sum/ num_batches
    std = np.sqrt(channels_squared_sum/num_batches - mean ** 2)

    return mean, std