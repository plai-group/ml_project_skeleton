import os
import sys

from ml_helpers import add_home, default_init
from src import assertions, data_handler, model_handler
import ml_helpers as mlh
from sacred import Experiment
import wandb
from time import sleep

# Use sacred for command line interface + hyperparams
# Use wandb for experiment tracking

ex = Experiment()
WANDB_PROJECT_NAME = 'my_project_name'
if '--unobserved' in sys.argv:
    os.environ['WANDB_MODE'] = 'dryrun'

# Put all hyperparameters + paths in my_config().
# and more complex data objects in init()

@ex.config
def my_config():
    # paths
    home_dir = '.' # required by job submitter, don't modify
    artifact_dir = './artifacts/' # required by job submitter, don't modify

    # relative to home_dir
    data_dir = '/data'
    weights_file = '/assets/example.csv'

    # Hyper params
    lr   = 0.001
    loss = 'adam'

    # Training settings
    epochs = 10
    seed   = 0
    cuda   = False


def init(config):
    # This gives dot access to all paths, hyperparameters, etc
    args = default_init(config)
    args.data_dir, args.weights_file = add_home(args.home_dir, args.data_dir,  args.weights_file)
    assertions.validate_hypers(args)

    args = mlh.detect_cuda(args)
    mlh.seed_all(args.seed)

    # get data
    args.data = data_handler.get_dataset(args)

    # get model
    args.model = model_handler.get_model(args)

    return args

# Main training loop
def train(args):
    loss = 0

    metric_logger = mlh.MetricLogger(wandb=wandb)

    for epoch in metric_logger.step(range(args.epochs)):
        loss += 1
        metric_logger.update(loss=loss, step=epoch)
        sleep(1)

    return loss


@ex.automain
def command_line_entry(_run,_config):
    wandb_run = wandb.init(project = WANDB_PROJECT_NAME,
                            config = _config,
                              tags = [_run.experiment_info['name']])
    args = init(_config)
    train(args)
