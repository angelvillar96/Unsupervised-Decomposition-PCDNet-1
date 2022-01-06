"""
Utils files for handling optuna studies
"""
import os
import json
import optuna
import torch

from CONFIG import OPTUNA
from lib.logger import print_


def create_optuna_values_file(exp_path):
    """ Creating optuna value file """
    exp_config = os.path.join(exp_path, "optuna_values.json")
    with open(exp_config, "w") as file:
        json.dump(OPTUNA, file)
    return


def load_optuna_values_file(exp_path):
    """ Creating optuna value file """
    exp_config = os.path.join(exp_path, "optuna_values.json")
    with open(exp_config) as file:
        data = json.load(file)
    return data


def load_optuna_study(exp_path, study_name=None):
    """ Loading and unpickeling an Optuna study """
    study_file = os.path.join(exp_path, "optuna.db")
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_file}")
    return study


def suggest_values(trial, exp_path, exp_params):
    """ Suggesting values for several hyper-parameters to optimize """
    values = load_optuna_values_file(exp_path)

    # TRAINING PARAMETERS
    batch_size = trial.suggest_int("BATCH", values["b_size_min"], values["b_size_max"])
    lr = trial.suggest_float("LR", values["lr_min"], values["lr_max"], log=True)
    scheduler_factor = trial.suggest_float("SCHEDULER_FACTOR", values["sch_factor_min"], values["sch_factor_max"])
    scheduler_patience = trial.suggest_int("SCHEDULER_PATIENCE", values["patience_min"], values["patience_max"])

    exp_params["training"]["batch_size"] = batch_size
    exp_params["training"]["lr"] = lr
    exp_params["training"]["lr_factor"] = scheduler_factor
    exp_params["training"]["patience"] = scheduler_patience

    # MODEL PARAMETERS
    num_protos = trial.suggest_int("NUM_PROTOS", values["num_protos_min"], values["num_protos_max"])
    num_objs = trial.suggest_int("NUM_OBJS", values["num_objs_min"], values["num_objs_max"])
    max_objs = trial.suggest_int("MAX_OBJS", values["max_objs_min"], values["max_objs_max"])
    rand_prob = trial.suggest_float("RAND_PROB", values["randomness_prob_min"], values["randomness_prob_max"])
    rand_iters = trial.suggest_int("RAND_ITERS", values["randomness_iters_min"], values["randomness_iters_max"])

    exp_params["model"]["PrototypeMatcher"]["num_protos"] = num_protos
    exp_params["model"]["PrototypeMatcher"]["randomness_prob"] = rand_prob
    exp_params["model"]["PrototypeMatcher"]["randomness_iters"] = rand_iters
    exp_params["model"]["PrototypeMatcher"]["num_objects"] = num_objs
    exp_params["model"]["PrototypeMatcher"]["max_objects"] = max_objs

    # LOSS PARAMETERS
    l1_reg = get_cat_or_log(trial=trial, min_val=values["lambda_l1_min"], max_val=values["lambda_l1_max"],
                            categorical=values["lambda_l1_cat"], name="L1_REG")
    l2_reg = get_cat_or_log(trial=trial, min_val=values["lambda_l2_min"], max_val=values["lambda_l2_max"],
                            categorical=values["lambda_l2_cat"], name="L2_REG")
    tv_reg = get_cat_or_log(trial=trial, min_val=values["lambda_tv_min"], max_val=values["lambda_tv_max"],
                            categorical=values["lambda_tv_cat"], name="TV_REG")

    exp_params["loss"]["lambda_l1"] = l1_reg
    exp_params["loss"]["lambda_l2"] = l2_reg
    exp_params["loss"]["lambda_tv"] = tv_reg

    return exp_params


def get_cat_or_log(trial, min_val, max_val, categorical, name):
    """
    Sampling a value from a categorical distribution with values logarithmically distributed,
    or directly sampling from a log-distribution
    """
    if min_val == 0 and max_val == 0:
        value = 0
    elif categorical:
        min_ = torch.log10(torch.tensor(min_val))
        max_ = torch.log10(torch.tensor(max_val))
        steps = (max_ - min_ + 1).int().item()
        val_list = torch.logspace(min_, max_, steps=steps).tolist()
        val_list = val_list + [0]  # adding value 0
        value = trial.suggest_categorical(name, val_list)
    else:
        value = trial.suggest_float(name, min_val, max_val, log=True)
    return value


def log_optuna_stats(study):
    """ """
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print_("Study statistics: ")
    print_(f"    Number of finished trials: {len(study.trials)}")
    print_(f"    Number of complete trials: {len(complete_trials)}")
    print_("")

    trial = study.best_trial
    print_(f"Best trial: Trial #{trial.number}")
    print_(f"  Value: {trial.value}")
    print_("  Params: ")
    for key, value in trial.params.items():
        print_(f"    {key}: {value}")

    return

#
