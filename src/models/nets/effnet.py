import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
import torchvision.models as models

from torchvision.models.efficientnet import Conv2dNormActivation, _efficientnet_conf

import pytorch_lightning as pl

from omegaconf import OmegaConf

class EfficientNet(pl.LightningModule):
    def __init__(self, 
                # general
                num_classes: int = None, 
                pretrained: bool = None, 
                params:dict = None,
                **kwargs):
        super(EfficientNet, self).__init__()

        params = OmegaConf.create(params)

        # Load pre-trained model
        self.model = models.efficientnet_b7(pretrained=pretrained)

        self.model.features[0] = self._load_pretrained_weights(self.model.features[0])

        # Replace the classifier with a new one for our task
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=params.dropout, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # todo: remove metrics from model and deal in callbacks
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)   
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)   

        # todo: remove this and use default 'hyper_params'
        self.save_hyperparameters()

    def _load_pretrained_weights(self, first_layer):
        "Load pretrained weights based on number of input channels"

        from functools import partial

        inverted_residual_setting, _ = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
        
        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        new_layer = Conv2dNormActivation(
                1, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), activation_layer=nn.SiLU
            )
        
        # todo fix this
        # # we take the sum
        # new_layer.weight.data = first_layer.weight.data.sum(dim=1, keepdim=True)
        # # and divide by the number of input channels
        # new_layer.weight.data /= getattr(first_layer, 'in_channels')
        return new_layer
    
    def forward(self, x):
        # Forward pass through the model
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
        # Use Adam optimizer with default parameters
        optimizer = torch.optim.Adam(self.parameters())

        return optimizer
