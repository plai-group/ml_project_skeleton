#!/usr/bin/env python
from job_submitter import submit
from pathlib import Path

project_path = Path(".").cwd() # don't modify

job_options = {
       "gpu": True,
       "hrs": 5,
       "cpu": 1,
       "mem": "12400M",
       "partition": 'plai',
       "env": 'ml3', # specify virtualenv, see static.py
       'account':'rrg-kevinlb' # for compute canada
}

default_args = {
    'resampling_method'    : ['reg', 'mult', 'stop'],  # will be iterated over
    "seed"                 : range(1, 5), # will be iterated over
    "epsilon"              : 0.5,
    "resampling_neff"      : 0.5,
    "scaling"              : 0.9,
    "convergence_threshold": 1e-3,
    "initial_lr"           : 0.0001,
    "decay"                : 0.75,
    "decay_steps"          : 250,
    "max_iter"             : 1000,
    "n_iter"               : 2000, # can be single value
    "n_particles"          : [500], # or within a list
    "data_dir"             : '/ubc/cs/research/fwood/vadmas/dev/projects/active/neurips2021/filterflow_neurips_2021/data',
    "file_name"            : ['jsb.pkl'],
}


submit(default_args,
        "jsb_reproduce_smaller_lr_larger_decay_steps",
        project_path,
        script_name='main.py',
        singularity='/ubc/cs/research/fwood/vadmas/containers/tensorflow_base.sif', # optional
        **job_options)
