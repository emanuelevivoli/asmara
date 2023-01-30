#!/usr/bin/env python
# coding: utf-8

import torch
import pytorch_lightning as pl

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.data.datasets.holo_data import LandmineDataset
from src.models.networks.resnet import ResNet50

from src.utils.data import chack_task_classes
from src.utils.spec import CLASS_NUMBER

import logging
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='configs', config_name='default')
def train(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.
    
    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))

    assert cfg.data.dataset != None, "Please specify a dataset in the config file. [holograms, inversion]"
    assert cfg.data.task != None, "Please specify a task in the config file. [bin, tri, fine-grain]"
    
    # We recover the original path of the dataset:
    datapath = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.paths[f'{cfg.data.dataset}'])
    metapath = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.paths.splits)

    # Load data
    train_data = LandmineDataset(data_path=datapath, 
                                meta_path=metapath, 
                                sample_rate=cfg.data.sample_rate, 
                                fold=cfg.data.folds.train, 
                                remove=cfg.data.remove)
    
    val_data = LandmineDataset(data_path=datapath,
                               meta_path=metapath, 
                               sample_rate=cfg.data.sample_rate, 
                               fold=cfg.data.folds.val, 
                               remove=cfg.data.remove)
    
    test_data = LandmineDataset(data_path=datapath, 
                                meta_path=metapath, 
                                sample_rate=cfg.data.sample_rate, 
                                fold=cfg.data.folds.test,
                                remove=cfg.data.remove)

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size)

    # check if train_loader is empty and batch size is correct
    if len(train_loader) == 0: raise ValueError("Train loader is empty. Check the configuration file.")
    batch = next(iter(train_loader))
    if len(batch) != 2: raise ValueError("The train loader is not returning the expected number of values. Check the configuration file.")

    pl.seed_everything(cfg.seed)

    # calculate the number of classes from the task
    # with a mapping from task to number of classes
    cfg.model.num_classes = CLASS_NUMBER[cfg.data.task]

    # check that data.task and model.num_classes are consistent
    chack_task_classes(cfg)

    # Initialize the network
    audionet = ResNet50(**cfg.model)
    # popolate the config trainer with configurations
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(audionet, train_loader, val_loader)

    # Test the network
    trainer.test(audionet, test_loader)

if __name__ == "__main__":
    train()
