import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision


import torch
import torch.nn as nn
import torchvision.models as models

        
class ResNet50(pl.LightningModule):

    def __init__(self, num_classes: int = 10, pretrained: bool = True, **kwargs):
        super(ResNet50, self).__init__()
        
        # to avoid warnings
        if pretrained: weights = 'ResNet50_Weights.DEFAULT'
        else: weights = None

        self.model = models.resnet50(weights=weights)
        self.model.conv1 = self._load_pretrained_weights(self.model.conv1)
        
        # Add a new fully-connected layer with the correct number of output classes
        self.fc = nn.Linear(in_features=1000, out_features=num_classes)

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define the accuracy metric
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _load_pretrained_weights(self, previous_layer):
        "Load pretrained weights based on number of input channels"
        # make a copy of the first layer
        new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
        # and divide by the number of input channels
        new_layer.weight.data /= getattr(previous_layer, 'in_channels')
        return new_layer

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
        logs = {"loss": loss, "acc": self.accuracy(logits, labels)}
        return {"loss": loss, "log": logs}

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
        logs = {"val_loss": loss, "val_acc": self.accuracy(logits, labels)}
        return {"val_loss": loss, "log": logs}
    
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
        logs = {"test_loss": loss, "test_acc": self.accuracy(logits, labels)}
        return {"test_loss": loss, "log": logs}

    def configure_optimizers(self):
        # Use SGD with a learning rate of 0.001 and momentum of 0
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.875)
        return optimizer