import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

class ResNet50(pl.LightningModule):

    def __init__(self, num_classes: int = 10):
        super(ResNet50, self).__init__()
        
        # Load a pre-trained ResNet50 model and remove the fully-connected layer
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Add a new fully-connected layer with the correct number of output classes
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

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