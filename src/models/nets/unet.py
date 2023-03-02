import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F

from omegaconf import OmegaConf

import pytorch_lightning as pl
import torchmetrics

import logging
logger = logging.getLogger(__name__)

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
        in_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        embed_dim: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
    """

    def __init__(self,
                # general
                num_classes: int = None, 
                pretrained: bool = None, 
                params: dict = None,
                **kwargs):

        params = OmegaConf.create(params)
        
        self.opt = kwargs.get('opt', None)

        if params.num_layers < 1:
            raise ValueError(f"params.num_layers = {params.num_layers}, expected: params.num_layers > 0")

        super().__init__()
        self.num_layers = params.num_layers

        layers = [DoubleConv(params.in_channels, params.embed_dim)]

        feats = params.embed_dim
        for _ in range(self.num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(self.num_layers - 1):
            layers.append(Up(feats, feats // 2, params.bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

        self.fc1 = nn.Linear(in_features = params.image_size*params.image_size*num_classes, out_features = params.hidden_dim)
        self.fc2 = nn.Linear(in_features = params.hidden_dim, out_features = num_classes)

        if pretrained:
            logger.error("Loading pretrained weights")
            self.load_state(s = 'unet_model_best.pth')
            # Linear_head map the patch embeddings to the number of classes
            self.model.linear_head = self.__set_head(params.embed_dim, params.hidden_dim, num_classes)
        else:
            logger.info("Training from scratch")

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define the accuracy metric
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers :-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        x = self.layers[-1](xi[-1])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # Get the input and labels from the batch
        inputs, labels = batch

        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate the loss and accuracy
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)

        # Log the loss and accuracy
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)

        logs = {"loss": loss, "acc": acc}
        return logs
    
    def validation_step(self, batch, batch_idx):
        # Get the input and labels from the batch
        inputs, labels = batch

        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate the loss and accuracy
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)

        # Log the loss and accuracy
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)

        logs = {"loss": loss, "acc": acc}
        return logs
    
    def test_step(self, batch, batch_idx):
        # Get the input and labels from the batch
        inputs, labels = batch
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate the loss and accuracy
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)

        # Log the loss and accuracy
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, sync_dist=True)

        logs = {"loss": loss, "acc": acc}
        return logs
        
    def configure_optimizers(self):
        if self.opt != None:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.opt.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters())
            
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
