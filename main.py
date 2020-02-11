import sacred
import numpy as np
from types import SimpleNamespace
from src.utils import seed_all, print_settings
from src import db, assertions
from sacred import Experiment
ex = db.init(Experiment())

# Put all hyperparameters + paths in my_config().
# Can handle basic python objects (strings, dicts, etc)
# https://sacred.readthedocs.io/en/stable/configuration.html
# More complex data objects (tensors, numpy arrays, pandas df)
# should be initialized in init()

@ex.config
def my_config():
    # paths
    model_dir = './models'
    data_dir = './data'

    # Hyper params
    lr   = 0.001
    loss = 'adam'

    # Training settings
    epochs = 10
    cuda   = False


def init(seed, config, _run):
    # This gives dot access to all paths, hyperparameters, etc
    args = SimpleNamespace(**config)
    assertions.validate_hypers(args)
    args.data_path = assertions.validate_dataset_path(args)

    # Seed everything
    seed_all(seed)
    args.seed = seed

    # This gives global access to sacred's '_run' object without having to capture functions
    args._run = _run

    # Other init stuff here (cuda, etc)
    return args


# Send metric to database
@ex.capture
def log_scalar(_run=None, **kwargs):
    assert "step" in kwargs, 'Step must be included in kwargs'
    step = kwargs.pop('step')

    for k, v in kwargs.items():
        _run.log_scalar(k, float(v), step)

    loss_string = " ".join(("{}: {:.4f}".format(*i) for i in kwargs.items()))
    print(f"Epoch: {step} - {loss_string}")


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

