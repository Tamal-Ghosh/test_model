import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Locks random seeds for python, numpy, and PyTorch (both CPU and CUDA)
    to ensure identical outputs for every run.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    # Force deterministic operations in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility seed locked to: {seed}")

def get_device():
    """
    Returns the best available device: GPU (cuda) or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
