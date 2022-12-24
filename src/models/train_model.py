#!/usr/bin/env python
# coding: utf-8

from src.utils.const import *
from utils.data import task_manager, get_holo_noise

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
logger = logging.getLogger(__name__)

class HoloDataset(torch.utils.data.Dataset):

    def __init__(self, metapath: Path = BASEPATH / Path('data/processed/meta'),
                       hologramspath: Path = BASEPATH / Path('data/interim/holograms'),
                       task: str = '2-class',
                       only_real: bool = True, 
                       remove: List = []) -> None:

        self.metapath = metapath
        # tasks that can be conducted are:
        # - '2-class'     - classification [mine / clutter]
        # - '3-class'   - classification [mine / clutter / arch]
        # - 'fg-class'  - fine-grain classification [vs50, wood-cylinder, ...]
        self.task = task
        self.only_real = only_real

        self.indoorpath = hologramspath / Path('indoor')
        self.outdoorpath = hologramspath / Path('outdoor')

        self.csv = pd.read_csv(metapath / Path('mix.csv'))
        # filter out some
        self.csv = self.csv[~self.csv['in_id'].isin(remove)]
        
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        in_holo = np.load(self.indoorpath / f"{row['in_file_name']}.npy")
        in_holo = torch.from_numpy(in_holo).double()
        
        holo_noise = get_holo_noise(self.indoorpath, row['in_file_name'], row['in_id'], in_holo.shape)
        holo_noise = torch.from_numpy(holo_noise).double()

        out_holo = np.load(self.outdoorpath / f"{row['out_file_name']}.npy")
        out_holo = torch.from_numpy(out_holo).double()

        print(in_holo.shape , holo_noise.shape, out_holo.shape)
        if out_holo.shape[-1] == 61:
            return None, None
        mix_holo =  ((in_holo - holo_noise) + out_holo)

        tasks = ['2-class','3-class','fg-class']

        if self.task in tasks:
            label = task_manager(self.task, row)
        else:
            raise ValueError(f'task {self.task} is not supported !')
        
        mix = torch.real(mix_holo) if self.only_real else mix_holo
        mix = mix.unsqueeze(0)
        mix = torch.nan_to_num(mix).T
        return mix.float(), label

    def __sizeof__(self) -> int:
        return len(self.csv)
    
    def __len__(self) -> int:
        return len(self.csv)



class UNet(pl.LightningModule):
    """Pytorch Lightning implementation of U-Net.

    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])

    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = functional.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.
    
    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))
    
    # We recover the original path of the dataset:
    path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    # Load data
    train_data = ESC50Dataset(path=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.train_folds)
    val_data = ESC50Dataset(path=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.val_folds)
    test_data = ESC50Dataset(path=path, sample_rate=cfg.data.sample_rate, folds=cfg.data.test_folds)

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size)

    pl.seed_everything(cfg.seed)

    # Initialize the network
    audionet = AudioNet(cfg.model)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(audionet, train_loader, val_loader)

if __name__ == "__main__":
    train()
