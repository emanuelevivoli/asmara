import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.HoloMineDataModule import HoloMineDataModule
from models.ResNet import ResNet50

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = HoloMineDataModule(batch_size=cfg.batch_size, split_type=cfg.split_type, transform=cfg.transform, shuffle=cfg.shuffle, num_workers=cfg.num_workers, classes=cfg.classes, interps=cfg.interps, inversion=cfg.inversion, real_out=cfg.real_out)
    net = cfg.net
    if net == 'resnet50':
        model = ResNet50(num_classes=cfg.classes, lr=cfg.lr)
    #TODO: model possibly unbound

    # elif net == 'resnext':
    #     model = ResNeXt.ResNetXt50(num_classes=cfg.classes, lr=cfg.lr)
    # elif net == 'vit':
    #     model = EfficientNet.EfficientNetV2(num_classes = cfg.classes, lr=cfg.lr)

    logger = TensorBoardLogger(save_dir=Path(__file__).parent.resolve() / "logs" / cfg.net, name = cfg.exp_name, log_graph = True)

    #TODO: make eary_stop_callback configurable with hydra config
    #early_stop_callback = EarlyStopping(monitor='val_loss', patience=cfg.patience, mode='min')

    #TODO: make checkpoint_callback configurable with hydra config
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        mode='min'
    )

    #TODO: trainer settings needs to be tested
    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], fast_dev_run = cfg.test_run)
    # trainer = pl.Trainer(accelerator='gpu', devices=[int(cfg.gpu)], strategy='ddp', logger=logger, profiler="simple", callbacks=[checkpoint_callback], fast_dev_run = cfg.test_run)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
