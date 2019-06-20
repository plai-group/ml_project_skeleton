import sacred
import numpy as np
from types import SimpleNamespace
from src.utils import seed_all, print_settings
from src import db
ex = db.init()

@ex.config
def my_config():
    # paths
    data_path = ''
    save_path = ''

    # Hyper params
    lr   = 0.001
    loss = 'adam'

    # Training settings
    epochs = 10
    cuda   = False

    # Assertions
    assert loss in ["adam"]

def init(seed, config, _run):
    # This gives dot access to all paths, hyperparameters, etc
    args = SimpleNamespace(**config)

    # Seed everything
    seed_all(seed)
    args.seed = seed

    # Print run settings
    print_settings(args)

    # This gives global access to sacred's '_run' object without having to capture functions
    args._run = _run

    # Other init stuff here (cuda, etc)
    return args


# Send metric to server
@ex.capture
def log_scalar(name, scalar, step=None, _run=None):
    assert np.isscalar(scalar) # Tensors, numpy arrays, etc wont work
    if step is not None:
        print("Epoch: {} - {}: {}".format(step, name, scalar))
        _run.log_scalar(name, scalar, step)
    else:
        print("{}: {}".format(name, scalar))
        _run.log_scalar(name, scalar)

# Main training loop
def train(args):
    loss = 0
    for epoch in range(args.epochs):
        log_scalar("loss", loss, step=epoch)
        loss += 1
    return loss

@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)
    result = train(args)
    # Whatever is returned from @ex.automain will be stored in the 'result' column in omniboard
    return result

