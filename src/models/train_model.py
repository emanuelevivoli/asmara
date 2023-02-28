#!/usr/bin/env python
# coding: utf-8

import torch
import pytorch_lightning as pl

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.utils.data import read_scans
from src.data.datasets.holo_data import HoloDataset
from src.models.networks.resnet import ResNet50

import logging
logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.
    
    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))
    
    # We recover the original path of the dataset:
    path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    indoor_dir = cfg.data.indoor_dir
    outdoor_dir = cfg.data.outdoor_dir
    indoor_scans, outdoor_scans = read_scans(indoor_dir, outdoor_dir)

    indoor_scans = torch.stack(indoor_scans)
    outdoor_scans = torch.stack(outdoor_scans)

    weights = [0.9 if i < len(outdoor_scans) else 0.1 for i in range(len(landmine_dataset))]
    sampler = WeightedRandomSampler(weights, len(weights))

    dataset = LandmineDataset(indoor_scans, outdoor_scans)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Load data
    train_data = LandimeDataset(procpath=procpath, interpath=interpath, task=task, sample_rate=cfg.data.sample_rate, folds=cfg.data.train_folds)
    val_data = LandmineDataset(metapath=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.val_folds)
    test_data = LandmineDataset(metapath=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.test_folds)

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size)

    pl.seed_everything(cfg.seed)

    # Initialize the network
    audionet = ResNet50(cfg.model)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(audionet, train_loader, val_loader)

if __name__ == "__main__":
    train()
