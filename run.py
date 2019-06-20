import numpy as np
from pprint import pprint
import losses
import itertools
from sacred import Experiment
from sacred.observers import MongoObserver
from pathlib import Path
import pickle
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import util
import torch
from data import get_data_loader

from types import SimpleNamespace

import db
ex = db.init()

@ex.config
def my_config():
    # paths
    data_path = '/ubc/cs/research/fwood/vadmas/dev/continuous_vae/data/data.pkl'
    save_path = '/ubc/cs/research/fwood/vadmas/dev/continuous_vae/trained_models'

    # Model
    architecture             = 'non_linear'
    loss                     = 'vae'
    num_stochastic_layers    = 1
    num_deterministic_layers = 2
    integration              = 'left'
    model_type               = 'continuous'
    cuda                     = True
    num_deterministic_layers = 0 if architecture == 'linear' else num_deterministic_layers

    # Hyper
    log_beta_min   = -1.09
    partition_type = "log"
    K  = 5
    S  = 10
    lr = 0.001
    test_K = 20

    # Training
    epochs = 5000
    batch_size = 1000
    test_batch_size = 1
    save_frequency  = int(epochs/5)
    valid_frequency = 10 if epochs < 100 else int(epochs/20)
    valid_S = 10
    test_S = 5000

    # Assertions
    assert architecture in ["linear", "non_linear"]
    assert partition_type in ['log', 'linear']
    assert loss in ['vae', 'iwae', 'thermo']
    assert integration in ['left', 'right', 'trap']
    assert model_type in ['continuous', 'discrete']

    # Name
    model_name = "{}.{}.{}.{}.{}".format(architecture, loss, num_stochastic_layers, num_deterministic_layers, S)
    if loss == 'thermo':
        model_name = "{}.{}.{}.{}".format(model_name, integration, log_beta_min, K)


@ex.capture
def log_scalar(name, scalar, step=None, _run=None):
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()

    if step is not None:
        print("Epoch: {} - {}: {}".format(step, name, scalar))
        _run.log_scalar(name, scalar, step)
    else:
        print("{}: {}".format(name, scalar))
        _run.log_scalar(name, scalar)


@ex.capture
def save_model(generative_model, inference_network, epoch, args, _run=None):
    inference_network_path = Path(args.save_path) / "{}_{}_{}.model".format(args.model_name, "inference_network", epoch)
    generative_model_path  = Path(args.save_path) / "{}_{}_{}.model".format(args.model_name, "generative_model", epoch)

    print("Saving {}".format(inference_network_path))
    print("Saving {}".format(generative_model_path))

    torch.save(inference_network.state_dict(), inference_network_path)
    torch.save(generative_model.state_dict(), generative_model_path)

    _run.add_artifact(inference_network_path)
    _run.add_artifact(generative_model_path)


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat independently."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init(seed, config, _run):
    seed_all(seed)
    args = SimpleNamespace(**config)
    args.seed = seed
    pprint(args.__dict__)

    args._run = _run
    losses_dict = {
        'vae':losses.get_vae_loss,
        'iwae':losses.get_iwae_loss,
        'thermo':losses.get_thermo_loss
    }

    args.loss = losses_dict[args.loss]


    args.partition = util.get_partition(args.S, args.partition_type, args.log_beta_min, args.device)
    return args


def train(args):
    # read data
    with args._run.open_resource(args.data_path, 'rb') as file_handle:
        data = pickle.load(file_handle)

    train_image = data['train_image']
    train_label = data['train_label']
    test_image  = data['test_image']
    test_label  = data['test_label']

    train_data_loader = get_data_loader(train_image, args.batch_size, args.device)
    test_data_loader  = get_data_loader(test_image, args.test_batch_size, args.device)

    import ipdb; ipdb.set_trace()
    train_obs_mean = torch.tensor(np.mean(train_image, axis=0), device=args.device, dtype=torch.float)
    generative_model, inference_network = util.init_models(train_obs_mean, args)

    parameters = itertools.chain.from_iterable([x.parameters() for x in [generative_model, inference_network]])
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    train_loss_epoch = []

    for epoch in range(args.epochs):
        running_loss = []
        epoch_train_elbo = 0

        for idx, data in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss, elbo = args.loss(generative_model, inference_network, data, args, args.valid_S)
            loss.backward()
            optimizer.step()
            epoch_train_elbo += elbo.item()

        # ---- end for --------
        epoch_train_elbo = epoch_train_elbo / len(train_data_loader)
        log_scalar("train.elbo", epoch_train_elbo, epoch)

        if (epoch != 0) and ((epoch % args.save_frequency) == 0):
            save_model(generative_model, inference_network, epoch, args)

        if (args.valid_frequency != 0)   and \
           ((epoch == (args.epochs - 1)) or ((epoch % args.valid_frequency) == 0)):
            test_elbo = 0
            with torch.no_grad():
                for idx, data in enumerate(test_data_loader):
                    _, elbo = args.loss(generative_model, inference_network, data, args, args.test_S)
                    test_elbo += elbo.item()

            test_elbo = test_elbo / len(test_data_loader)
            log_scalar("test.elbo", test_elbo, epoch)

        # ------ end of training loop ---------
    save_model(generative_model, inference_network, args.epochs, args)
    return test_elbo

@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)
    result = train(args)
    return result

