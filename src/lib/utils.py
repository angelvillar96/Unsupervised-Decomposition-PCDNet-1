"""
Utils methods for bunch of purposes
"""

import os
import json
import random
import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt

from lib.logger import log_function
from CONFIG import CONFIG


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = CONFIG["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def clear_cmd():
    """Clearning command line window"""
    os.system('cls' if os.name=='nt' else 'clear')
    return


@log_function
def create_directory(dir_path, dir_name=None):
    """
    Creating a folder in given path.
    """
    if(dir_name is not None):
        dir_path = os.path.join(dir_path, dir_name)
    if(not os.path.exists(dir_path)):
        os.makedirs(dir_path)
    return


def timestamp():
    """ Obtaining the current timestamp in an human-readable way """

    timestamp = str(datetime.datetime.now()).split('.')[0] \
                                            .replace(' ', '_') \
                                            .replace(':', '-')

    return timestamp


def log_architecture(model, exp_path, fname="model_architecture.txt"):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"

    savepath = os.path.join(exp_path, fname)
    with open(savepath, "w") as f:
        f.write("")
    with open(savepath, "a") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write( str(model) )

    return


def rgb2gray(img):
    """ Converting an RGB image into grayscale """
    r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
    img = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # img = 0.33*r + 0.33*g + 0.33 * b
    return img
