import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from omegaconf import DictConfig
from torch.nn import functional as F

from src.utils.data import task_manager
from src.utils.spec import TASKS

import logging
logger = logging.getLogger(__name__)

class LandmineDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: Path,
                       meta_path: Path,
                       fold: str,
                       cfg: DictConfig,
                       transform=None) -> None:

        self.data_path = data_path
        self.cfg = cfg
        
        self.fold = fold
        
        self.csv = pd.read_csv(meta_path / f'{fold}.csv')
        self.csv = self.csv[~self.csv['in_id'].isin(self.cfg.data.remove)]
        self.task = cfg.data.task

        self.out_type = cfg.data.out_type
        self.transform = transform
        
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        suffix = '_inv' if self.cfg.data.dataset == 'inversions' else ''
        data = np.load(self.data_path / f"{row['mix_name']}{suffix}.npy", allow_pickle=True)
        data = torch.from_numpy(data).cfloat()
        
        # todo: remove noise, if present, based on the  file
        # in_noise = np.load(self.data_path / f"{row['out_file_name']}.npy")
        # in_noise = torch.from_numpy(in_noise).double()
        # data = (data - in_noise)

        # data.shape should be torch.Size([52, 62])
        if self.cfg.data.source != 'interps' and \
            (data.shape[0] != 52 or data.shape[1] != 62):
            
            real_data = torch.view_as_real(data) # [52, 61]
            real_data = real_data.permute(2, 0, 1).unsqueeze(0) # [1, 2, 52, 61]
            real_data = F.interpolate(real_data, size=(52, 62), mode='nearest') # [1, 2, 52, 62]
            real_data = real_data.squeeze(0).permute(1, 2, 0).contiguous() # [52, 62, 2]
            data = torch.view_as_complex(real_data) # [52, 62]
        
        # get correct label based on self.task property
        if self.task in TASKS:
            label = task_manager(self.task, row)
        else:
            raise ValueError(f'task {self.task} is not supported !')
        
        # if needed (property self.out_type) extract real component
        
        if self.out_type=='real':
            mix = torch.real(data).unsqueeze(0)
            mix = torch.nan_to_num(mix)
        else:
            mix = data.unsqueeze(0)
    
        mix = mix.permute(*torch.arange(mix.ndim - 1, -1, -1))


        # mix.shape = torch.Size([62, 52, 1])
        # check dmeensions
        if self.cfg.data.source != 'interps':
            if mix.shape[0] != 62:
                raise ValueError(f'mix.shape[0] != 62, but {mix.shape[0]}')
            if mix.shape[1] != 52:
                raise ValueError(f'mix.shape[1] != 52, but {mix.shape[1]}')
            if mix.shape[2] != 1:
                raise ValueError(f'mix.shape[2] != 1, but {mix.shape[2]}')
        
        # reshape the mix to have channels first
        mix = mix.permute(2, 0, 1)

        if self.transform:
            mix = self.transform(mix)
        
        return mix.float(), label

    def __sizeof__(self) -> int:
        return len(self.csv)
    
    def __len__(self) -> int:
        return len(self.csv)
