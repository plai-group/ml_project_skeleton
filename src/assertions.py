# Assertions
def validate_hypers(args):
    assert args.loss in ['adam']
    # Add an assertion everytime you catch yourself making a silly hyperparameter mistake so it doesn't happen again

def validate_dataset_path(args):
    # place conditionals here
    data_path = args.data_dir + 'data.pkl'
    return data_path
