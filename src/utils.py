import random
import torch
import numpy as np
from pprint import pprint
from types import SimpleNamespace

def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_settings(args):
    print("==============SETTINGS================")
    pprint(args.__dict__)
    print("--------------------------------------")

