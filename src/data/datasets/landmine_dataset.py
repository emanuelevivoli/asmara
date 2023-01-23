import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class LandmineDataset(Dataset):
    def __init__(self, indoor_scans, outdoor_scans):
        self.indoor_scans = indoor_scans
        self.outdoor_scans = outdoor_scans
        
    def __len__(self):
        return len(self.outdoor_scans) + len(self.indoor_scans)

    def __getitem__(self, idx):
        if idx < len(self.outdoor_scans):
            return self.outdoor_scans[idx]
        else:
            idx -= len(self.outdoor_scans)
            scan_indoor = self.indoor_scans[idx]
            scan_outdoor = random.choice(self.outdoor_scans)
            scan = scan_indoor + scan_outdoor
            return scan
