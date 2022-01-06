"""
Setting up the model, optimizers, loss functions, loading/saving parameters, ...
"""

import os
import torch

from lib.logger import print_, log_function
from lib.utils import create_directory
import models


@log_function
def setup_model(model_params, dataset_name=""):
    """
    Loading the model
    """
    model_name = model_params["model_name"]

    if(model_name == "PCDNet"):
        proto_params = model_params["PrototypeMatcher"]
        num_objs = proto_params["num_objects"]
        proto_params["max_objects"] = proto_params.get("max_objects", num_objs)

        # use_color = True if dataset_name in ["", "Tetrominoes", "Cars",
        #                                      "CarsTop", "CarsSide"] else False
        # channels = 3 if dataset_name in ["Tetrominoes", "Cars",
        #                                  "CarsTop", "CarsSide"] else 1
        use_color = True if dataset_name in ["", "Tetrominoes"] else False
        channels = 3 if dataset_name in ["", "Tetrominoes"] else 1

        model = models.DecompModel(
                num_protos=proto_params["num_protos"],
                num_objects=proto_params["num_objects"],
                max_objects=proto_params["max_objects"],
                proto_size=proto_params["proto_size"],
                background_size=proto_params["background_size"],
                mode=proto_params["proto_init"],
                channels=channels,
                colorTf=use_color,
                randomness=proto_params["proto_randomness"],
                randomness_prob=proto_params["randomness_prob"],
                randomness_iters=proto_params["randomness_iters"],
                background=proto_params["background"],
                use_empty=proto_params["use_empty"]
            )
    else:
        raise NotImplementedError()

    return model


@log_function
def save_checkpoint(model, optimizer, scheduler, epoch, meta, exp_path, finished=False,
                    savedir="models", savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler
    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    epoch: integer
        current epoch number
    meta: dictionary
        some other metadata to dump inot the checkpoint
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = f"checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    create_directory(exp_path, savedir)
    savepath = os.path.join(exp_path, savedir, checkpoint_name)

    scheduler_data = "" if scheduler is None else scheduler.state_dict()
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data,
            "meta": meta
            }, savepath)
    return


@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler
    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """

    if(checkpoint_path is None):
        return model

    # loading model to either cpu or cpu
    if(map_cpu):
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)
    # loading model parameters. Try catch is used to allow different dicts
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        model.load_state_dict(checkpoint)

    # returning only the model for transfer learning or returning also optimizer state
    # for continuing training procedure
    if(only_model):
        return model

    optimizer, scheduler = None, None
    if "optimizer" in kwargs and kwargs["optimizer"] is not None:
        optimizer = kwargs["optimizer"]
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if "scheduler" in kwargs and kwargs["scheduler"] is not None:
        scheduler = kwargs["scheduler"]
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if "meta" in checkpoint:
        meta = checkpoint["meta"]

    epoch = checkpoint["epoch"]

    return model, optimizer, scheduler, epoch, meta


@log_function
def setup_optimizer(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters
    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model
    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """

    lr = exp_params["training"]["lr"]
    lr_factor = exp_params["training"]["lr_factor"]
    patience = exp_params["training"]["patience"]
    momentum = exp_params["training"]["momentum"]
    optimizer = exp_params["training"]["optimizer"]
    # weight_decay = exp_params["training"]["weight_decay"]
    weight_decay = exp_params["training"].get("weight_decay", 0)
    nesterov = exp_params["training"]["nesterov"]
    scheduler = exp_params["training"]["scheduler"]

    # SGD-based optimizer
    if(optimizer == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=weight_decay)

    # LR-scheduler
    if(scheduler == "plateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience,
                                                               factor=lr_factor, min_lr=1e-8,
                                                               mode="min", verbose=True)
    elif(scheduler == "step"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_factor,
                                                    step_size=patience, verbose=True)
    else:
        scheduler = None

    return optimizer, scheduler


def update_scheduler(scheduler, exp_params, control_metric):
    """
    Updating the learning rate scheduler
    """
    scheduler_type = exp_params["training"]["scheduler"]
    if(scheduler_type == "plateau"):
        scheduler.step(control_metric)
    elif(scheduler_type == "step"):
        scheduler.step()

    return


#
