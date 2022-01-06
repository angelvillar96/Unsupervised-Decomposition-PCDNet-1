"""
Global configurations
"""

import os

CONFIG = {
    "random_seed": 13,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "..", "..", "datasets"),
        # "experiments_path": os.path.join(os.getcwd(), "..", "models"),
        "experiments_path": os.path.join(os.getcwd(), "..", "experiments"),
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
            "proto_randomness": True,
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
        "lambda_l1": 0,           # multiplyier for the L1 regularization term
        "lambda_l2": 0,           # multiplyier for the L2 regularization term
        "lambda_tv": 0,           # multiplyier for the total variation regularization term
    },
    "training": {  # training related parameters
        "num_epochs": 100,      # number of epochs to train for
        "save_frequency": 3,    # saving a checkpoint after these eoochs ()
        "log_frequency": 100,   # logging stats after this amount of updates
        "batch_size": 64,
        "lr": 3e-3,
        "lr_factor": 1,
        "weight_decay": 0.,
        "momentum": 0,
        "nesterov": False,
        "patience": 5,
        "optimizer": "adam",
        "scheduler": "step"
    }
}

# ranges for optuna hyper-param study
OPTUNA = {
    "lr_min": 1e-5,  # training parameters
    "lr_max": 1e-1,
    "b_size_min": 4,
    "b_size_max": 64,
    "sch_factor_min": 1e-2,
    "sch_factor_max": 0.9,
    "patience_min": 1,
    "patience_max": 3,
    "weight_decay_min": 0,
    "weight_decay_max": 0,
    "num_protos_min": 19,  # model parameters
    "num_protos_max": 19,
    "num_objs_min": 3,
    "num_objs_max": 3,
    "max_objs_min": 3,
    "max_objs_max": 3,
    "randomness_prob_min": 0,
    "randomness_prob_max": 1,
    "randomness_iters_min": 0,
    "randomness_iters_max": 1000,
    "lambda_l1_min": 0,  # loss parameters
    "lambda_l1_max": 0,
    "lambda_l1_cat": True,
    "lambda_l2_min": 1e-6,
    "lambda_l2_max": 1e-1,
    "lambda_l2_cat": True,
    "lambda_tv_min": 1e-6,
    "lambda_tv_max": 1e-1,
    "lambda_tv_cat": True,
}
