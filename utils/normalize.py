import torch

def normalize(data, max):
    return (data / max) * 2 - 1

def unnormalize(data, max):
    return (data + 1) / 2 * max

def log_normalize(data):
    return torch.log(data + 1)
