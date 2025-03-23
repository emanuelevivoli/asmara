import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
import hydra
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.HoloMineDataModule import HoloMineDataModule
from models.ResNet import ResNet50
from models.ResNeXt import ResNeXt50

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = HoloMineDataModule(batch_size=cfg.batch_size, split_type=cfg.split_type, transform=cfg.transform, shuffle=cfg.shuffle, num_workers=cfg.num_workers, classes=cfg.classes, interps=cfg.interps, inversion=cfg.inversion, real_out=cfg.real_out)
    
    net = cfg.get("net", None)  # Safely get net, default to None if missing
    if net == "resnet50":
        model = ResNet50(num_classes=cfg.classes, lr=cfg.lr)
    elif net == "resnext50":
        model = ResNeXt50(num_classes=cfg.classes, lr=cfg.lr)
    else:
        raise ValueError(f"Invalid or missing net parameter: {net}. Choose from ['resnet50', 'resnext50'].")

    logger = TensorBoardLogger(save_dir=Path(__file__).parent.resolve() / "logs" / cfg.net, name = cfg.exp_name, log_graph = True)

    #TODO: make eary_stop_callback configurable with hydra config
    #early_stop_callback = EarlyStopping(monitor='val_loss', patience=cfg.patience, mode='min')

    #TODO: make checkpoint_callback configurable with hydra config
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        mode='min'
    )

    if type(cfg.accelerator) == str:
        accelerator = cfg.accelerator
        devices = 'auto'
    else:
        accelerator = 'gpu'
        devices = cfg.accelerator


    trainer = pl.Trainer(accelerator=accelerator, devices = devices, logger=logger, callbacks=[checkpoint_callback], fast_dev_run = cfg.test_run)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
