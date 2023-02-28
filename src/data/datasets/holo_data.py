import torch
import pandas as pd
import numpy as np

from pathlib import Path
from typing import List

from src.utils.data import task_manager, get_holo_noise
from src.utils.const import BASEPATH

class HoloDataset(torch.utils.data.Dataset):

    def __init__(self, procspath: Path = BASEPATH / Path('data/processed'),
                       interpath: Path = BASEPATH / Path('data/interim'),
                       task: str = '2-class',
                       only_real: bool = True, 
                       remove: List = []) -> None:

        self.metapath = procspath / Path('meta')
        self.hologramspath = interpath / Path('holograms')
        # tasks that can be conducted are:
        # - '2-class'     - classification [mine / clutter]
        # - '3-class'   - classification [mine / clutter / arch]
        # - 'fg-class'  - fine-grain classification [vs50, wood-cylinder, ...]
        self.task = task
        self.only_real = only_real

        self.indoorpath = self.hologramspath / Path('indoor')
        self.outdoorpath = self.hologramspath / Path('outdoor')

        self.csv = pd.read_csv(self.metapath / Path('mix.csv'))
        # filter out some
        self.csv = self.csv[~self.csv['in_id'].isin(remove)]
        
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        in_holo = np.load(self.indoorpath / f"{row['in_file_name']}.npy")
        in_holo = torch.from_numpy(in_holo).double()
        
        holo_noise = get_holo_noise(self.indoorpath, row['in_file_name'], row['in_id'], in_holo.shape)
        holo_noise = torch.from_numpy(holo_noise).double()

        out_holo = np.load(self.outdoorpath / f"{row['out_file_name']}.npy")
        out_holo = torch.from_numpy(out_holo).double()

        print(in_holo.shape , holo_noise.shape, out_holo.shape)
        if out_holo.shape[-1] == 61:
            return None, None
        mix_holo =  ((in_holo - holo_noise) + out_holo)

        tasks = ['2-class','3-class','fg-class']

        if self.task in tasks:
            label = task_manager(self.task, row)
        else:
            raise ValueError(f'task {self.task} is not supported !')
        
        mix = torch.real(mix_holo) if self.only_real else mix_holo
        mix = mix.unsqueeze(0)
        mix = torch.nan_to_num(mix).T
        return mix.float(), label

    def __sizeof__(self) -> int:
        return len(self.csv)
    
    def __len__(self) -> int:
        return len(self.csv)

