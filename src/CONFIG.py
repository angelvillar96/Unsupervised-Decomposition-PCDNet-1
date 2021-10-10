"""
Global configurations
"""

import os

CONFIG = {
    "random_seed": 13,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "..", "..", "datasets"),
        "experiments_path": os.path.join(os.getcwd(), "..", "models"),
        }
}


DEFAULTS = {
    "dataset": {
        "dataset_name": "Tetrominoes",
        "shuffle_train": True,
        "shuffle_eval": False
    },
    "model": {
        "model_name": "PCDNet",  # type of model to use
        "PrototypeMatcher": {                  # parameters for prototype learning
            "num_protos": 19,
            "num_objects": 3,
            "max_objects": 3,
            "proto_init": "constant",
            "proto_size": 25,
            "background_size": 35,
            "template_reg": 0,
            "proto_randomness": False,
            "randomness_prob": 0.2,
            "randomness_iters": 800,
            "background": False,
            "init_background": False,
            "use_empty": False
        }
    },
    "loss": {
        "loss_recons": "mse",    # loss functuions
        "lambda_recons": 1,      # mulitplying factors for the loss functions
        "template_reg": False,   # regularization that penalizes non-empty templates
        "lambda_reg": 0,         # multiplyier for the regularization term
        "lambda_l1": 0           # multiplyier for the L1 regularization term
    },
    "training": {  # training related parameters
        "num_epochs": 100,      # number of epochs to train for
        "save_frequency": 3,    # saving a checkpoint after these eoochs ()
        "log_frequency": 100,   # logging stats after this amount of updates
        "batch_size": 64,
        "lr": 3e-3,
        "lr_factor": 1,
        "momentum": 0,
        "nesterov": False,
        "patience": 5,
        "optimizer": "adam",
        "scheduler": None
    }
}
