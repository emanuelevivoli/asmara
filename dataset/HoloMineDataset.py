import torch
from torch.nn import functional as F

import numpy as np
import pandas as pd

from pathlib import Path

def task_manager(classes: int, row) -> torch.Tensor:
    if classes == 1:
        return torch.tensor([1 if row['in_name'] == 'mine' else 0], dtype=torch.float32)
    elif classes == 3:
        return torch.tensor([1 if row['in_name'] == 'mine' else 0 if row['in_name'] == 'archeology' else 2], dtype=torch.float32)
    else:
        return torch.tensor([int(row['in_id']) - 1], dtype=torch.float32)

class HoloMineDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        classes: int = 1,
        interps: bool = True,
        inversion: bool = False,
        real_out: bool = True,
    ) -> None:

        self.classes = classes
        self.interps = interps
        self.inversion = inversion
        self.real_out = real_out

        data_path = Path(__file__).parent.resolve() / "data/processed_data"
        if self.interps:
            data_path = data_path / "interpolated"
        else:
            data_path = data_path / "standard"

        self.meta_path = data_path / "meta"

        if self.inversion:
            self.data_path = data_path / "inversions"
        else:
            self.data_path = data_path / "holograms"

        self.csv = pd.read_csv(self.meta_path / "mix.csv")

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        suffix = "_inv" if self.inversion else ""
        data = np.load(
            self.data_path / f"{row['mix_name']}{suffix}.npy", allow_pickle=True
        )
        data = torch.from_numpy(data).cfloat()

        if not self.interps and (
            data.shape[0] != 52 or data.shape[1] != 62
        ):

            real_data = torch.view_as_real(data)  # [52, 61]
            real_data = real_data.permute(2, 0, 1).unsqueeze(0)  # [1, 2, 52, 61]
            real_data = F.interpolate(
                real_data, size=(52, 62), mode="nearest"
            )  # [1, 2, 52, 62]
            real_data = (
                real_data.squeeze(0).permute(1, 2, 0).contiguous()
            )  # [52, 62, 2]
            data = torch.view_as_complex(real_data)  # [52, 62]

        label = task_manager(self.classes, row)

        if self.real_out:
            mix = torch.real(data).unsqueeze(0)
            mix = torch.nan_to_num(mix)
        else:
            mix = data.unsqueeze(0)

        mix = mix.permute(*torch.arange(mix.ndim - 1, -1, -1))

        if not self.interps:
            if mix.shape[0] != 62:
                raise ValueError(f"mix.shape[0] != 62, but {mix.shape[0]}")
            if mix.shape[1] != 52:
                raise ValueError(f"mix.shape[1] != 52, but {mix.shape[1]}")
            if mix.shape[2] != 1:
                raise ValueError(f"mix.shape[2] != 1, but {mix.shape[2]}")

        # reshape the mix to have channels first
        if not self.inversion:
            mix = mix.permute(2, 0, 1)
        else:
            mix = mix.permute(3, 2, 0, 1)

        return mix.float(), label

    def __sizeof__(self) -> int:
        return len(self.csv)

    def __len__(self) -> int:
        return len(self.csv)
