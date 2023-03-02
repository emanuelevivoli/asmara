import os
import shutil
from omegaconf import OmegaConf

import torch
import torchmetrics
import torch.nn as nn

import pytorch_lightning as pl
from vit_pytorch import SimpleViT as sViT

import logging
logger = logging.getLogger(__name__)

class SimpleViT(pl.LightningModule):
    """ Simple ViT model for classification tasks.
    """

    def __init__(self, 
                # general
                num_classes: int = None, 
                pretrained: bool = None, 
                params:dict = None,
                **kwargs):

        super(SimpleViT, self).__init__()

        params = OmegaConf.create(params)

        self.opt = kwargs.get('opt', None)

        self.model = sViT(
            num_classes = num_classes,
            image_size = params.image_size,
            patch_size = params.patch_size,
            channels = params.in_channels,
            dim = params.embed_dim,
            depth = params.num_layers,
            heads = params.num_heads,
            mlp_dim = params.mlp_dim,
        )
        
        if pretrained:
            logger.info("Loading pretrained weights")
            self.load_state(s = 'vit_model_best.pth')
            # Linear_head map the patch embeddings to the number of classes
            self.model.linear_head = self.__set_head(params.embed_dim, params.hidden_dim, num_classes)
        else:
            logger.info("Training from scratch")
    
        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define the accuracy metric
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def __set_head(self, dim, hidden_dim, num_classes, norm_layer=True, dropout=0.2):
        head = [nn.Linear(dim, hidden_dim)]
        if norm_layer: head.append(nn.LayerNorm(hidden_dim))
        head.append(nn.ReLU(inplace=True))
        if dropout > 0: head.append(nn.Dropout(dropout))
        head.append(nn.Linear(hidden_dim, num_classes))
        
        return nn.Sequential(*head)

    def forward(self, x):
        x = self.model(x)
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