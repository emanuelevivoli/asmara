import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn as nn
import torchvision.models as models

        
class ResNet50(pl.LightningModule):
    """ ResNet50 model for classification tasks.

    We have different ways of interacting with the model, in case we want to adapt it to
    different input sizes. We have the following options:
    
    A.  (different number of channels)
        To change the first convolutional layer of the model to accept the new input size. 
        This is done by loading the pretrained weights of the first convolutional layer, 
        and then replacing the first convolutional layer with a new one that has the correct
        number of input channels.

    B.  (different image size)
        To change the average pooling layer of the model to accept the new input size. 
        This is done by replacing the average pooling layer with a new one that has the correct
        kernel size.

    C.  (different image size)
        To change the fully-connected layer of the model to accept the new input size.
        This is done by replacing the fully-connected layer with a new one that has the correct
        number of input features.

    D.  (different number of classes)
        To change the output layer of the model to accept the new input size.
        This is done by replacing the output layer with a new one that has the correct
        number of output classes.

    In our case we have different number of channels (A), different image size (B and C),
    and different number of classes (D).

    We also have the option to change the input image size, to match the default input size of 224x224:

    E.  (different image size)
        To change the input image size of the model to accept the new input size.
        This is done by replacing the input image size with a new one that has the correct
        input size.
    """

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
        # Use SGD with a learning rate of 0.001 and momentum of 0
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.875)
        return optimizer