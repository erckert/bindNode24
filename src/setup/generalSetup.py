import numpy as np
import torch
import random


def seed_all(seed=10):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'