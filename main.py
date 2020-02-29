import os
import sys
import sacred
import numpy as np
from types import SimpleNamespace
from src.utils import seed_all
from src import assertions
from sacred import Experiment
import wandb

ex = Experiment()
if '--unobserved' in sys.argv:
    os.environ['WANDB_MODE'] = 'dryrun'

# Use sacred for command line interface + hyperparams
# Use wandb for experiment tracking monitoring

# Put all hyperparameters + paths in my_config().
# and more complex data objects in init()

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

    wandb.init(project="my-test-project",
               config=config,
               tags=[_run.experiment_info['name']])

    args.data_path = assertions.validate_dataset_path(args)

    # Seed everything
    seed_all(seed)
    args.seed = seed

    # Other init stuff here (cuda, etc)
    return args


def log_scalar(**kwargs):
    wandb.log(kwargs)
    loss_string = " ".join(("{}: {:.4f}".format(*i) for i in kwargs.items()))
    print(f"{loss_string}")


# Main training loop
def train(args):
    loss = 0
    for epoch in range(args.epochs):
        log_scalar(loss=loss, step=epoch)
        loss += 1
    return loss

@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)
    result = train(args)

    return result

