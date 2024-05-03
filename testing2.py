import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

from pathlib import Path
from typing import List
from omegaconf import DictConfig
from torch.nn import functional as F

def loadFestoLog(FFESTO):
    opt = {
        "delimiter" : ";",
        "names" : ["DATE", "X", "Y", "Z"]
    }
    festoLog = pd.read_csv(FFESTO, **opt)
    festoLog.DATE = pd.to_datetime(festoLog.DATE,  unit="ms")

    festoLog.X = festoLog.X/1000
    festoLog.Y = festoLog.Y/1000
    festoLog.Z = festoLog.Z/1000
    
    DATE = festoLog
    return DATE

def loadPlutoLog(FPLUTO):
    opt = {
        "delimiter" : " ",
        "names" : ["DATE", "FREQUENCY_TX", "FREQUENCY_RX", "MOD_I", "PHASE_I", "MOD_Q", "PHASE_Q", "MOD", "PHASE"],
        "dtype" : {'FREQUENCY_TX': float, 'FREQUENCY_RX': float, 'MOD_I': float, 
                   'PHASE_I': float, 'MOD_Q': float, 'PHASE_Q': float, 'MOD': float, 'PHASE': float},
        "parse_dates": ["DATE"],
        "date_parser": lambda x: pd.to_datetime(x, format="%Y/%m/%d-%H:%M:%S.%f"),
        "skipinitialspace": True,
        "skiprows": 1
    }

    plutoScan = pd.read_csv(FPLUTO, **opt)
    DATE = plutoScan
    return DATE

def unifyLog(festo_file_path, pluto_file_path):

    # definisco costanti
    NEIG = "none"
    INTM = "natural"
    TOFF = 0
    P1 = loadFestoLog(festo_file_path)
    co, ce = np.histogram(P1.Z, 1000)
    cx = np.argmax(co)
    H = ce[cx]*1000
    P2 = loadPlutoLog(pluto_file_path)
    P2.DATE = P2.DATE + pd.to_timedelta(TOFF * 1e-3, unit='s') #TOFF inutile dato che mantiene valore 0 per tutto il programma
    #verifico saturazione segnali
    if any((abs(P2.MOD_I * np.sin(P2.PHASE_I)) >= 2**11-1) | (abs(P2.MOD_I * np.cos(P2.PHASE_I)) >= 2**11-1)): 
        print('WARNING: signal I is saturated.')
    if any((abs(P2.MOD_Q * np.sin(P2.PHASE_Q)) >= 2**11-1)| (abs(P2.MOD_Q * np.cos(P2.PHASE_Q)) >= 2**11-1)):
        print('WARNING: signal Q is saturated.')
    #Index of PLUTO's data within FESTO's time window
    P_ix = np.where((P2.DATE >= min(P1.DATE)) & (P2.DATE <= max(P1.DATE)))
    P_F = P2.FREQUENCY_TX[P_ix[0]] #P_ix it's a list of arrays
    P_F_UNIQUE = np.unique(P_F)
    P_F_UNIQUE = np.sort(P_F_UNIQUE)
    #P_DATE = np.empty(len(P_ix[0]), dtype="datetime64[ns]")
    P_DATE = P2.DATE[P_ix[0]]
    P_MOD = P2.MOD[P_ix[0]]
    P_PHASE = P2.PHASE[P_ix[0]]

    #festo data
    F_X = P1.X
    F_Y = P1.Y
    F_DATE = P1.DATE

    FFMOD = []
    FFPHASE = []
    P_X = []
    P_Y = []
    for n in range(0, len(P_F_UNIQUE)):
        spline_interp_x = interp1d(F_DATE, F_X, kind='cubic', bounds_error=False, fill_value="extrapolate")
        spline_interp_y = interp1d(F_DATE, F_Y, kind='cubic', bounds_error=False, fill_value="extrapolate")
        P_X.append(spline_interp_x(P_DATE))
        P_Y.append(spline_interp_y(P_DATE))
    px = np.array(P_X[0])
    py = np.array(P_Y[0])

    # Definizione della griglia regolare
    xi = np.linspace(min(px), max(px), 67)
    yi = np.linspace(max(py), min(py), 54)

    # Interpolazione dei dati sparsi sulla griglia regolare
    FFMOD = griddata((px, py), P_MOD.values, (xi[None, :], yi[:, None]), method = "linear")
    FFPHASE = griddata((px, py), P_PHASE.values, (xi[None, :], yi[:, None]), method = "linear")

    # zi conterrà i valori interpolati sulla griglia regolare
    FFMOD[np.isnan(FFMOD)] = np.min(FFMOD)
    FFMOD[np.isnan(FFPHASE)] = np.min(FFPHASE)
    FFMOD = FFMOD[1:-2,5:-1]  
    FFPHASE = FFPHASE[1:-2,5:-1]
    fourier_transform = FFMOD * np.exp(1j * FFPHASE)
    
    return fourier_transform

class LandmineDataset_bis(torch.utils.data.Dataset):
    def __init__(self, indoor_scans: Path, outdoor_scans: Path, mix_labels: Path):
        #self.indoor_scans = indoor_scans #for interpolation
        #self.outdoor_scans = outdoor_scans
        # self.csv = pd.read_csv(mix_labels)
        self.csv = pd.read_csv(mix_labels)
        self.indoor_scans = {file.stem: np.load(file) for file in indoor_scans.iterdir() if file.is_file()}
        self.outdoor_scans = {file.stem: np.load(file) for file in outdoor_scans.iterdir() if file.is_file()}

    
    def __len__(self):
        return len(self.outdoor_scans) + len(self.indoor_scans)
    
    def interpolate(self):
        # Per scorrere tutti i file nella cartella indoor_scans
        for indoor_file in self.indoor_scans.iterdir():
            if indoor_file.is_file():
                indoor = np.load(indoor_file)
                indoor = torch.from_numpy(indoor).cfloat()
                indoor = torch.view_as_real(indoor) # [52, 61]
                indoor = indoor.permute(2, 0, 1).unsqueeze(0) # [1, 2, 52, 61]
                indoor = F.interpolate(indoor, size=(64, 64), mode='nearest') # [1, 2, 52, 62]
                indoor = indoor.squeeze(0).permute(1, 2, 0).contiguous() # [52, 62, 2]
                indoor = torch.view_as_complex(indoor)
                np.save(f"/home/lbisognin/ASMARA/data/machine_learning_bis/interps/indoor/{indoor_file.name}", indoor)

        # Esempio di come scorrere i file in outdoor_scans se necessario
        for outdoor_file in self.outdoor_scans.iterdir():
            if outdoor_file.is_file():
                outdoor = np.load(outdoor_file)
                outdoor = torch.from_numpy(outdoor).cfloat()
                outdoor = torch.view_as_real(outdoor) # [52, 61]
                outdoor = outdoor.permute(2, 0, 1).unsqueeze(0) # [1, 2, 52, 61]
                outdoor = F.interpolate(outdoor, size=(64, 64), mode='nearest') # [1, 2, 52, 62]
                outdoor = outdoor.squeeze(0).permute(1, 2, 0).contiguous() # [52, 62, 2]
                outdoor = torch.view_as_complex(outdoor)
                np.save(f"/home/lbisognin/ASMARA/data/machine_learning_bis/interps/outdoor/{outdoor_file.name}", outdoor)
                
    def __getimage__(self, idx):
        row = self.csv.iloc[idx]
        indoor = self.indoor_scans[f"{row['in_file_name']}_holo"]
        plt.imshow(np.real(indoor), origin="lower", cmap="gray")
        plt.margins(0)
        plt.axis("off")
        plt.savefig(f'/home/breakfastea/projects/asmara/data_bis/interim/standard/images/{row["mix_name"]}.png', bbox_inches='tight', pad_inches=0)
        plt.show
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        indoor = self.indoor_scans[f"{row['in_file_name']}_holo"]
        outdoor = self.outdoor_scans[f"{row['out_file_name']}_holo"]
        alpha = 0.70
        fusion = alpha * indoor + (1-alpha) * outdoor
        np.save(f'/home/lbisognin/asmara-main/data/processed/holograms_worst/{row["mix_name"]}.npy', fusion)
        # plt.imshow(np.real(fusion) ,origin="lower", cmap="gray")
        # plt.margins(0)
        # plt.axis("off")
        # if row['in_name'] == 'mine':
        #     plt.savefig(f'/home/lbisognin/ASMARA/data/machine_learning_bis/mineclassify_bis/train/mine/{row["mix_name"]}.png', bbox_inches='tight', pad_inches=0)
        # else:
        #     plt.savefig(f'/home/lbisognin/ASMARA/data/machine_learning_bis/mineclassify_bis/train/other/{row["mix_name"]}.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
    
hologram = unifyLog('data/tests/TraceData_0229_1_A.csv', 'data/tests/plutoScan_0229_1_A.log')
plt.imshow(np.real(hologram), origin="lower", cmap="gray")
plt.margins(0)
plt.axis("off")
plt.savefig(f'/home/breakfastea/projects/asmara/data_bis/interim/standard/images/test.png', bbox_inches='tight', pad_inches=0)
plt.show

# dataset = LandmineDataset_bis(Path("/home/breakfastea/projects/asmara/data_bis/interim/standard/holograms/indoor"),
#                                Path("/home/breakfastea/projects/asmara/data_bis/interim/standard/holograms/outdoor"),
#                               Path("/home/breakfastea/projects/asmara/data/interim/meta/mix.csv"))

# for i in tqdm(range(0, 47632)):
#     dataset.__getitem__(i)
# # # #      #355


# dataset.__getimage__(0)
