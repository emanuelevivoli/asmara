import sys
import os
from torch.utils.data import random_split, DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HoloMineDataset import HoloMineDataset

class HoloMineDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, split_type: str = 'soil', transform: bool = False, shuffle:bool = True, num_workers: int = 1, classes: int = 1, interps:bool = True, inversion:bool = False, real_out:bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.full = None
        self.train = None
        self.val = None
        self.test = None
        self.data_split_done = False
        if transform:
            self.rotation_transform = transforms.Compose([
                transforms.RandomChoice([
                    transforms.RandomRotation(90),
                    transforms.RandomRotation(180),
                    transforms.RandomRotation(270),
                    transforms.RandomRotation(360)
                ]),
            ])
        else:
            self.rotation_transform = None

    def setup(self, stage=None):
        if not self.data_split_done:
            self.full = HoloMineDataset(classes=self.hparams.classes, interps=self.hparams.interps, inversion=self.hparams.inversion, real_out=self.hparams.real_out)
            if self.hparams.split_type == 'random':
                self.train, self.val, self.test = random_split(self.full, [0.8, 0.1, 0.1])
                self.data_split_done = True

            elif self.hparams.split_type == 'soil':
                file_names = self.full.csv["out_file_name"].values
                out_ids = set()

                for file_name in file_names:
                    out_id = file_name.split('_')[0]
                    out_ids.add(out_id)

                out_ids = list(out_ids)
                np.random.shuffle(out_ids)

                train_size = int(0.8 * len(out_ids))
                val_size = int(0.1 * len(out_ids))

                train_out_ids = out_ids[:train_size]
                val_out_ids = out_ids[train_size:train_size+val_size]
                test_out_ids = out_ids[train_size+val_size:]

                train_indices = []
                val_indices = []
                test_indices = []

                for i, file_name in enumerate(file_names):
                    out_id, rotation = file_name.split('_')
                    if out_id in train_out_ids:
                        train_indices.append(i)
                    elif out_id in val_out_ids:
                        val_indices.append(i)
                    elif out_id in test_out_ids:
                        test_indices.append(i)

                self.train = Subset(self.full, train_indices)
                self.train.dataset.transform = self.rotation_transform
                self.val = Subset(self.full, val_indices)
                self.test = Subset(self.full, test_indices)

                self.data_split_done = True

            elif self.hparams.split_type == 'pair':
                file_names = self.full.csv["out_file_name"].values
                unique_pairs = set()
                
                for file_name in file_names:
                    out_id, rotation = file_name.split('_')
                    unique_pairs.add((out_id, rotation))  # Collect unique pairs

                unique_pairs = list(unique_pairs)
                np.random.shuffle(unique_pairs)

                train_size = int(0.8 * len(unique_pairs))
                val_size = int(0.1 * len(unique_pairs))

                train_pairs = unique_pairs[:train_size]
                val_out_ids = unique_pairs[train_size:train_size+val_size]
                test_out_ids = unique_pairs[train_size+val_size:]

                train_indices = []
                val_indices = []
                test_indices = []

                for i, file_name in enumerate(file_names):
                    out_id, rotation = file_name.split('_')
                    if (out_id, rotation) in train_pairs:
                        train_indices.append(i)
                    elif (out_id, rotation) in val_out_ids:
                        val_indices.append(i)
                    elif (out_id, rotation) in test_out_ids:
                        test_indices.append(i)

                self.train = Subset(self.full, train_indices)
                self.train.dataset.transform = self.rotation_transform
                self.val = Subset(self.full, val_indices)
                self.test = Subset(self.full, test_indices)

                self.data_split_done = True

            else:
                raise ValueError(f"Invalid input: {self.hparams.split_type}, choose [random | soil | pair]")

    def train_dataloader(self):
        if not self.data_split_done:
            raise RuntimeError("setup() must be called before defining data loaders")
        return DataLoader(self.train, batch_size = self.hparams.batch_size, shuffle = self.hparams.shuffle, num_workers = self.hparams.num_workers, persistent_workers=True)

    def val_dataloader(self):
        if not self.data_split_done:
            raise RuntimeError("setup() must be called before defining data loaders")
        return DataLoader(self.val, batch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers, persistent_workers=True)

    def test_dataloader(self):
        if not self.data_split_done:
            raise RuntimeError("setup() must be called before defining data loaders")
        return DataLoader(self.test, batch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers, persistent_workers=True)
