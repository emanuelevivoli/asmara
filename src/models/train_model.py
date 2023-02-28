#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.data.datasets.holo_data import LandmineDataset

from src.utils.spec import CLASS_NUMBER
from src.utils.data import check_task_classes
from src.utils.model import instantiate_from_config


import logging
logger = logging.getLogger(__name__)

wandb_logger = WandbLogger()

@hydra.main(version_base=None, config_path='../config', config_name='default')
def train(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.
    
    assert cfg.data.dataset != None, "Please specify a dataset in the config file. [holograms, inversion]"
    assert cfg.data.task != None, "Please specify a task in the config file. [bin, tri, fine-grain]"
    assert cfg.model.name != None, "Please specify model name. [ResNet50, SimpleViT, UNet]"

    model_cfg = OmegaConf.load(f"src/config/models/{cfg.model.name}.yaml")
    
    cfg.model = OmegaConf.merge(cfg.model, model_cfg)

    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))

    # calculate the number of classes from the task
    # with a mapping from task to number of classes
    cfg.model.num_classes = CLASS_NUMBER[cfg.data.task]
    # check that data.task and model.num_classes are consistent
    check_task_classes(cfg)

    # We recover the original path of the dataset:
    datapath = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.paths[f'{cfg.data.dataset}'])
    logging.info(f"Using {cfg.data.dataset} dataset from {datapath}")

    metapath = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.paths.splits)

    # Define the transformations to apply to the data
    if cfg.data.transforms == 'default':
        logging.info("Using transformations -> [60x60]")
        transform = transforms.Compose([
            transforms.Resize((60, 60)),  # rescale to 60x60
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # normalize the data
        ])
    else:
        logging.info("Using NO transformations")
        transform = None

    # create train, validation, and test datasets
    train_data = LandmineDataset( datapath, metapath, 'train', cfg, transform )
    val_data = LandmineDataset( datapath, metapath, 'val', cfg, transform )
    test_data = LandmineDataset( datapath, metapath, 'test', cfg, transform )
    
    
    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True, **cfg.dataset)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size, **cfg.dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size, **cfg.dataset)

    # check if train_loader is empty and batch size is correct
    if len(train_loader) == 0: raise ValueError("Train loader is empty. Check the configuration file.")
    batch = next(iter(train_loader))
    if len(batch) != 2: raise ValueError("The train loader is not returning the expected number of values. Check the configuration file.")

    pl.seed_everything(cfg.seed)

    # Initialize the network
    model = instantiate_from_config(cfg.model)

    # popolate the config trainer with configurations
    trainer = pl.Trainer(
        **cfg.trainer, 
        # when strategy:'ddp' and find_unused_parameters:False, 
        strategy= DDPStrategy(find_unused_parameters=False) 
            if cfg.custom_trainer.strategy == 'ddp' and 
                cfg.custom_trainer.find_unused_parameters == False 
            else 'ddp',
        logger = wandb_logger,
    )
    
    trainer.fit(model, train_loader, val_loader)

    # Test the network
    trainer.test(model, test_loader)

if __name__ == "__main__":
    train()
