#!/usr/bin/env python
from job_submitter import submit
from pathlib import Path

project_path = Path(".").cwd()
job_options = {
    "gpu": True,
    "hrs": 2,
    "mem": "12400M",
    "queue": 'gpu',
    "sleep_time":1.0
}

continuous = {
    "S": [5, 10, 50],
    "learning_task": ['lin_reg', 'log_reg', 'gmm', 'gmm_amortized'],
    "loss": ['tvo'],
    "schedule" : ['log', 'random_uniform'],
    "lr": [0.01],
    "K": [3, 10, 50],
    "seed": [0, 1],
    "valid_S": 5000
}

submit(continuous, "log_vs_random", project_path, **job_options)
