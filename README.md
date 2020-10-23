An opinionated machine learning project skeleton using sacred + wandb. This uses Sacred for its command line interface, where any hyper added to `my_config()` becomes available on the commandline via:

`python main.py with my_hyper='hello world'`.

See sacred's [documentation](https://sacred.readthedocs.io/en/stable/command_line.html) for more information.  The `import ml_helpers` in `main.py` refers to [my ml helper functions](https://github.com/vmasrani/machine_learning_helpers), and the latex directory is structured to hide as much of the tedious boilerplate code as possible. 
