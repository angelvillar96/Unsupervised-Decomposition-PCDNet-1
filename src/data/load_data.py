"""
Methods for loading specific datasets, fitting data loaders and other
"""

from torch.utils.data import DataLoader
from data import Tetrominoes, CarsDataset, AttariDataset
from CONFIG import CONFIG


def load_data(exp_params, split="train", transform=None):
    """
    Loading a dataset given the parameters

    Args:
    -----
    exp_params: dictionary
        dict with the experiment specific parameters
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')
    transform: Torch Transforms
        Compose of torchvision transforms to apply to the data

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    in_channels: integer
        number of channels in the dataset samples (e.g. 1 for B&W, 3 for RGB)
    """
    DATA_PATH = CONFIG["paths"]["data_path"]

    # reading dataset parameters from the configuration file
    available_dbs = ["Tetrominoes", "VMDS", "Cars", "CarsTop", "CarsSide", "AtariSpace"]
    dataset_name = exp_params["dataset"]["dataset_name"]

    if(dataset_name == "Tetrominoes"):
        dataset = Tetrominoes(
            path=DATA_PATH,
            split=split
        )
    elif(dataset_name == "SpritesMOT"):
        dataset = SpriteDataset(
            path=DATA_PATH,
            mode=split,
            rgb=True
        )
    elif(dataset_name == "VMDS"):
        dataset = SpriteDataset(
            path=DATA_PATH,
            mode=split,
            rgb=True,
            dataset_class="vmds"
        )
    elif(dataset_name == "Cars" or dataset_name == "CarsTop"):
        # dataset = CarsDataset(split=split, video="top", rgb=True)
        dataset = CarsDataset(split=split, video="top", rgb=False)
    elif(dataset_name == "CarsSide"):
        dataset = CarsDataset(split=split, video="side", rgb=True)
    elif(dataset_name == "AtariSpace"):
        dataset = AttariDataset(mode=split, game="SpaceInvaders", rgb=False)
    else:
        raise NotImplementedError(f"""ERROR! Dataset'{dataset_name}' is not available.
            Please use one of the following: {available_dbs}...""")

    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    return data_loader
