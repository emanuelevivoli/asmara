import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics.classification
import torchvision.models as models
import torchmetrics

class ResNet50(pl.LightningModule):
    def __init__(self, num_classes: int = 1, lr: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, 1, 64, 64)

        self.model = models.resnet50()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(2048, num_classes)

        if num_classes == 1:
            self.loss = nn.BCEWithLogitsLoss()
            self.accuracy = torchmetrics.classification.BinaryAccuracy()
            self.f1 = torchmetrics.classification.BinaryF1Score()
            self.prec = torchmetrics.classification.BinaryPrecision()
            self.rec = torchmetrics.classification.BinaryRecall()
        else:
            self.loss = nn.CrossEntropyLoss()
            self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)   
            self.f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)
            self.prec = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes)
            self.rec = torchmetrics.classification.MulticlassRecall(num_classes=num_classes)

    def forward(self, inputs):
        out = self.model(inputs)
        return out

    def training_step(self, batch):
        inputs, labels = batch
        output = self(inputs)
        loss = self.loss(output, labels)
        acc= self.accuracy(output, labels)
        f1 = self.f1(output, labels)
        prec = self.prec(output, labels)
        rec = self.rec(output, labels)

        logs = {"train_loss": loss, "train_acc": acc, "train_f1": f1, "train_prec": prec, "train_rec": rec}
        self.log_dict(logs, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        output = self(inputs)
        loss = self.loss(output, labels)
        acc= self.accuracy(output, labels)
        f1 = self.f1(output, labels)
        prec = self.prec(output, labels)
        rec = self.rec(output, labels)

        logs = {"val_loss": loss, "val_acc": acc, "val_f1": f1, "val_prec": prec, "val_rec": rec}
        self.log_dict(logs, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch):
        inputs, labels = batch
        output = self(inputs)
        loss = self.loss(output, labels)
        acc= self.accuracy(output, labels)
        f1 = self.f1(output, labels)
        prec = self.prec(output, labels)
        rec = self.rec(output, labels)

        logs = {"test_loss": loss, "test_acc": acc, "test_f1": f1, "test_prec": prec, "test_rec": rec}
        self.log_dict(logs, prog_bar=True, logger=False, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.lr)

    def extract_features(self, x):
        features = nn.Sequential(*list(self.model.children())[:-1])(x)
        return features.view(features.size(0), -1)
