import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from torch.nn import functional as F

from src.utils.data import task_manager
from src.utils.spec import TASKS

import logging
logger = logging.getLogger(__name__)
class LandmineDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: Path,
                       meta_path: Path,
                       sample_rate: float = 0.2,
                       task: str = 'bin',
                       out_type: str = 'real',
                       fold: str = 'train',
                       remove: List = []) -> None:

        self.data_path = data_path
        # tasks that can be conducted are:
        # - 'bin'   - classification [mine / clutter]
        # - 'tri'   - classification [mine / clutter / arch]
        # - 'fg-class'  - fine-grain classification [vs50, wood-cylinder, ...]
        self.sample_rate = sample_rate
        self.task = task
        self.out_type = out_type

        # filter out some
        self.csv = pd.read_csv(meta_path / f'{fold}.csv')
        self.csv = self.csv[~self.csv['in_id'].isin(remove)]
        
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        data = np.load(self.data_path / f"{row['mix_name']}.npy")
        data = torch.from_numpy(data).cfloat()
        
        # todo: remove noise, if present, based on the  file
        # in_noise = np.load(self.data_path / f"{row['out_file_name']}.npy")
        # in_noise = torch.from_numpy(in_noise).double()
        # data = (data - in_noise)

        # data.shape = torch.Size([62, 52])
        # check dmeensions
        if data.shape[0] != 52 or data.shape[1] != 62:
            logger.warning(f'data.shape = {data.shape} (should be 52x62)')
            # [52, 61]
            real_data = torch.view_as_real(data)
            # [52, 61, 2] HWC to [2, 52, 61] CHW (channel, height, width)
            real_data = real_data.permute(2, 0, 1)
            # [1, 2, 52, 61] BCHW
            real_data = real_data.unsqueeze(0)
            # The input dimensions are interpreted in the form: 
            # mini-batch x channels x [optional depth] x [optional height] x width
            real_data = F.interpolate(real_data, size=(52, 62), mode='nearest')
            # [1, 2, 52, 62] BCHW to [2, 52, 62] CHW
            real_data = real_data.squeeze(0)
            # [2, 52, 62] CHW to [52, 62, 2] HWC
            real_data = real_data.permute(1, 2, 0).contiguous()
            # [52, 62, 2] HWC to [52, 62] complex64
            data = torch.view_as_complex(real_data)
        
        # get correct label based on self.task property
        if self.task in TASKS:
            label = task_manager(self.task, row)
        else:
            raise ValueError(f'task {self.task} is not supported !')
        
        # if needed (property self.out_type) extract real component
        mix = (torch.real(data) if self.out_type=='real' else data).unsqueeze(0)
        mix = torch.nan_to_num(mix)
        mix = mix.permute(*torch.arange(mix.ndim - 1, -1, -1))


        # mix.shape = torch.Size([62, 52, 1])
        # check dmeensions
        if mix.shape[0] != 62:
            raise ValueError(f'mix.shape[0] != 62, but {mix.shape[0]}')
        if mix.shape[1] != 52:
            raise ValueError(f'mix.shape[1] != 52, but {mix.shape[1]}')
        if mix.shape[2] != 1:
            raise ValueError(f'mix.shape[2] != 1, but {mix.shape[2]}')
        
        # reshape the mix to have channels first
        mix = mix.permute(2, 0, 1)
        
        return mix.float(), label

    def __sizeof__(self) -> int:
        return len(self.csv)
    
    def __len__(self) -> int:
        return len(self.csv)

