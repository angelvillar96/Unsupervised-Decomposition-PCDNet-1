"""
Methods for processing command line arguments
"""

import os
import argparse

from CONFIG import CONFIG


def create_experiment_arguments():
    """
    Processing arguments for 01_*
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Directory where the experiment" +\
                        "folder will be created", required=True, default="test_dir")
    args = parser.parse_args()

    return args


def get_directory_argument(get_checkpoint=False):
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory", required=True)
    parser.add_argument("--checkpoint", help="Relative path to the model to evaluate")
    args = parser.parse_args()

    exp_directory = args.exp_directory
    checkpoint = args.checkpoint
    exp_directory = process_experiment_directory_argument(exp_directory)

    if(checkpoint is not None):
        checkpoint = process_checkpoint(checkpoint, exp_directory)

    return exp_directory, checkpoint


def process_experiment_directory_argument(exp_directory):
    """
    Ensuring that the experiment directory argument exists
    and giving the full path if relative was detected
    """

    was_relative = False
    exp_path = CONFIG["paths"]["experiments_path"]
    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # making sure experiment directory exists
    if(not os.path.exists(exp_directory)):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()

    return exp_directory


def process_checkpoint(checkpoint, exp_dir):
    """ Ensuring that the given model checkpoint exists """
    path = os.path.join(exp_dir, "models", "PCDNet_models", checkpoint)
    assert os.path.exists(path), f"Checkpoint {checkpoint} does not exist"
    return checkpoint
