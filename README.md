# Unsupervised Image Decomposition with Phase-Correlation Networks


![PCDNet Illustration](resources/PCDNet.png "Illustration of PCDNet")

This respository contains the main codebase for the paper: *Unsupervised Image Decomposition with Phase-Correlation Networks*.
[[PDF](https://arxiv.org/abs/2110.03473)]

The repository allows to reproduce the experiments and results from the paper.


## Contents

 * [1. Getting Started](#getting-started)
 * [2. Directory Structure](#directory-structure)
 * [3. Quick Guide](#quick-guide)
 * [4. Citation](#citation)
 * [5. Contact](#contact)


 ## Getting Started

 To download the code, fork the repository or clone it using the following command:

 ```
   git clone git@github.com:angelvillar96/Unsupervised-Decomposition-PCDNet.git
 ```


 ### Prerequisites

 To get the repository running, you will need several python packages, e.g., Numpy, OpenCV PyTorch or Matplotlib.

 You can install them all easily and avoiding dependency issues by installing the conda environment file included in the repository. To do so, run the following command from the Conda Command Window or from a Terminal:

 ```shell
 $ conda env create -f environment.yml
 $ conda activate PCDNet
 ```

 *__Note__:* This step might take a few minutes



 ## Directory Structure

 The following tree diagram displays the detailed directory structure of the project. Some directory names and paths can be modified in the [CONFIG File](https://github.com/angelvillar96/Unsupervised-Decomposition-PCDNet/blob/master/src/CONFIG.py).

 ```
 Unsupervised-Decomposition-PCDNet
 |
 ├── resources/
 |
 ├── datasets/
 |   ├── AAD/  
 |   ├── Tetrominoes/  
 |   └── cars/  
 |
 ├── models/
 |
 ├── src/
 |   ├── data/  
 │   |── lib/
 │   |── models/
 |   |── notebooks/
 │   ├── 01_create_experiment.py
 │   ├── 02_train.py
 │   ├── 03_evaluate.py
 │   └── 04_generate_figures.py
 |
 ├── environment.yml
 └── README.md
 ```


Now, we give a short overview of the different directories:

- **resources/**: Images from the README.md file

- **datasets/**: Image datasets used in the paper. It can be downloaded from [here](https://www.dropbox.com/sh/qba6w3bfmdou7l4/AAAdG2DpDVGOMfsL-WXB3XzZa?dl=0)

- **models/**: Exemplary experiments and pretrained models

- **src/**: Code to reproduce the results from the paper.


## Quick Guide

In this section, we explain how to use the repository to reproduce the experiments from the paper.

##### Creating an Experiment

For creating an experiment, run
```shell
$ python 1_create_experiment.py [-h] -d EXP_DIRECTORY
```  

The instruction automatically generates a directory in the specified EXP_DIRECTORY, containing a *JSON* file with the experiment parameters and subdirectories for the models and plots.


##### Training and Evaluation

Once the experiment is initialized and the parameters from the *experiment_params.json* file are set, a PCDNet model can be trained running the following command.

```shell
$ CUDA_VISIBLE_DEVICES=0 python 02_train_dissentangle_model.py -d YOUR_EXP_DIRECTORY [--checkpoint CHECKPOINT]
```

Additionally, you can resume training from a checkpoint using the *--checkpoint* argument.

##### Generating Figures

Run the following command to generate several figures, including the learned object and mask prototypes, reconstructions, segmentations, and decompositions.

```shell
$ CUDA_VISIBLE_DEVICES=0 python 04_generate_figures.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the experiment directory
  --checkpoint CHECKPOINT
                        Relative path to the model to evaluate
```

For example:

```shell
$ CUDA_VISIBLE_DEVICES=0 python 04_generate_figures.py -d Tetrominoes --checkpoint tetrominoes_model.pth

```


<img src="resources/tetris.png" alt="Tetris masks and prototypes" width="50%"/><img src="resources/protos_atari.png" alt="Space Invaders Prototypes" width="50%"/>

<img src="resources/atari.png" alt="PCDNet reconstructions and segmentations on Space Invaders" width="95%"/>

<img src="resources/cars.png" alt="PCDNet reconstructions, prototypes and masks on the NGSIM dataset" width="95%"/>


## Citation
Please consider citing our paper if you find our work or our repository helpful.
```
@inproceedings{villar2021PCDNet,
  title={Unsupervised Image Decomposition with Phase-Correlation Networks},
  author={Villar-Corrales, A. and Behnke, S.},
  booktitle={International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP)},
  year={2022}
}
```


## Contact

This repository is maintained by [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php),

In case of any questions or problems regarding the project or repository, do not hesitate to contact the authors at villar@ais.uni-bonn.de.
