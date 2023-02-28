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
                # model
                image_size: int = 52, 
                patch_size: int = 16, 
                num_classes: int = 10, 
                dim: int = 1024, 
                depth: int = 6, 
                heads: int = 16, 
                mlp_dim: int = 2048, 
                channels:int = 1, 
                # general
                hidden_dim: int = 100,
                pretrained: bool = True, 
                **kwargs):

        super(SimpleViT, self).__init__()

        self.model = sViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            channels = channels
        )
        
        if pretrained:
            logger.info("Loading pretrained weights")
            self.load_state(s = 'vit_model_best.pth')
            # Linear_head map the patch embeddings to the number of classes
            self.model.linear_head = self.__set_head(dim, hidden_dim, num_classes)
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
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate the loss
        loss = self.loss_fn(logits, labels)
        
        # Log the loss and accuracy
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(logits, labels))

        logs = {"loss": loss, "acc": self.accuracy(logits, labels)}
        return logs

    def validation_step(self, batch, batch_idx):
        # Get the input and labels from the batch
        inputs, labels = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate the loss
        loss = self.loss_fn(logits, labels)
        
        # Log the loss and accuracy
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(logits, labels))

        logs = {"loss": loss, "acc": self.accuracy(logits, labels)}
        return logs

    def test_step(self, batch, batch_idx):
        # Get the input and labels from the batch
        inputs, labels = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate the loss
        loss = self.loss_fn(logits, labels)
        
        # Log the loss and accuracy
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy(logits, labels))

        logs = {"loss": loss, "acc": self.accuracy(logits, labels)}
        return logs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, momentum=0.875)
        return optimizer

    def save_state(self, state, is_best):
        out = os.path.join(CHECKPOINTS, f'{self.name}_checkpoint.pth')
        torch.save(state, out)
        if is_best:
            shutil.copyfile(out, os.path.join(CHECKPOINTS,f'{self.name}_model_best.pth'))
    
    def load_state(self, s):
        s = torch.load(os.path.join(CHECKPOINTS, s))
        self.model.load_state_dict(s['state_dict'])
    
    def resume(self, r):
        print(f"=> loading checkpoint '{r}'")
        c = torch.load(os.path.join(CHECKPOINTS, r))
        self.load_state(c['state_dict'])
        return c['epoch'], c['best_val_loss'], c['exit_counter'], c['optimizer'], c['best_val_f1']