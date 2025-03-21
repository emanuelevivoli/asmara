import os
import shutil
from omegaconf import OmegaConf

import torch
import torchmetrics
import torch.nn as nn

import pytorch_lightning as pl
from vit_pytorch.simple_vit_3d import SimpleViT as sViT3d

import logging
logger = logging.getLogger(__name__)

class SimpleViT3d(pl.LightningModule):
    """ Simple ViT model for classification tasks.
    """

    def __init__(self, 
                # general
                num_classes: int = None, 
                pretrained: bool = None, 
                params:dict = None,
                **kwargs):

        super(SimpleViT3d, self).__init__()

        params = OmegaConf.create(params)

        self.opt = kwargs.get('opt', None)
        
        self.model = sViT3d(
            num_classes = num_classes,
            image_size = params.image_size,
            image_patch_size = params.image_patch_size,
            frames = params.frames,
            frame_patch_size = params.frame_patch_size,
            channels = params.in_channels,
            dim = params.embed_dim,
            depth = params.num_layers,
            heads = params.num_heads,
            mlp_dim = params.mlp_dim,
        )
        
        if pretrained:
            logger.info("Loading pretrained weights not supported for SimpleViT3d")
            raise NotImplementedError
        else:
            logger.info("Training from scratch")
    
        # Define the loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()

        # todo: remove metrics from model and deal in callbacks
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)   

        # todo: remove this and use default 'hyper_params'
        self.save_hyperparameters()


    def forward(self, x):
        #todo: remove the first channel fo images 
        x = x[:, :, 1:, :, :]
        x = self.model(x)
        return x
    
    def step_wrapper(self, batch, batch_idx, mode):

        # Get the input and labels from the batch
        inputs, labels = batch
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate the loss and accuracy
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)

        # Log the loss and accuracy
        self.log(f"{mode}_loss", loss, sync_dist=True)
        self.log(f"{mode}_acc", acc, sync_dist=True)
        self.log(f"{mode}_f1", f1, sync_dist=True)

        logs = {"loss": loss, "acc": acc, "f1": f1}
        return logs
    
    def training_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, mode='val')

    def test_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, mode='test')

    def configure_optimizers(self):
        
        if self.opt != None:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.opt.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters())
            
        return optimizer