# ASMARA

## Overview
ASMARA is a landmine detection project developed in collaboration with MICC, UNIFI, and NATO. The project focuses on advanced data processing and deep learning techniques to improve landmine detection capabilities.

## Project Structure
```
.
├── dataset
│   ├── data
│   ├── scripts
│   ├── HoloMineDataModule.py
│   ├── HoloMineDataset.py
│   └── README.md
├── docs
├── experiments
│   ├── configs
│   │   └── config.yaml
│   ├── logs
│   ├── models
│   └── train.py
├── notebooks
├── outputs
├── environment.yaml
├── LICENSE
├── make_dataset.py
└── README.md
```

## Installation
Before proceeding with the setup, ensure that **MiniConda** and **MATLAB** are already installed on your system. MATLAB is required for certain functionalities, but if you do not plan to use it, modify `environment.yaml` to exclude MATLAB-related dependencies to ensure a smooth environment setup.

To create the Conda environment, run:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate asmara
```

## Dataset Preparation
By default, the dataset generation process excludes 3D inversion files due to their high computational cost. However, if you need these files, you can generate them using the `--full` flag:
```bash
python make_dataset.py --full
```
Once completed, the dataset will be fully available for use.

For testing or modifying dataset generation, or to generate only specific parts, use:
```bash
python ./dataset/scripts/make_interm.py --help
```
```bash
python ./dataset/scripts/make_processed.py --help
```

## Training Configuration and Execution
The training settings are defined in `config.yaml`. It is recommended to check [Hydra](https://hydra.cc/) to understand how to manage multiple configuration files or how to override settings via the command line.

The experiment section of this repository leverages **PyTorch Lightning** for data loading, model definition, and training management. If you prefer not to use PyTorch Lightning, the dataset is also available as a standard PyTorch `Dataset` in `HoloMineDataset.py`.

To start training with the specified configuration:
```bash
python ./experiments/train.py
```

## Monitoring Training with TensorBoard
By default, **TensorBoard** is used for logging and visualization. You can launch TensorBoard and access the provided local URL:
```bash
tensorboard --logdir ./experiments/logs/
```

## Contributing
If you are interested in contributing to the project, please review the README files within the **dataset** and **experiments** directories. These documents provide detailed insights into the project's structure and workflow, enabling more effective contributions.

For any additional information or to report issues, please refer to the documentation or open an issue on the project repository.
The dataset generation process has been fully tested and should work without issue, the experiment setup has been tested only on ResNet and ResNeXt, using interpolated .npy files for binary classification. Any other configuration may require modifications to the model definitions and adjustments to the methods within HoloMineDataModule. 

