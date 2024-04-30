import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
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
    xi = np.linspace(min(px), max(px), 62)
    yi = np.linspace(max(py), min(py), 52)

    # Interpolazione dei dati sparsi sulla griglia regolare
    FFMOD = griddata((px, py), P_MOD.values, (xi[None, :], yi[:, None]), method = "linear")
    FFPHASE = griddata((px, py), P_PHASE.values, (xi[None, :], yi[:, None]), method = "linear")
    print("test")
    # zi conterrà i valori interpolati sulla griglia regolare
    FFMOD[np.isnan(FFMOD)] = np.min(FFMOD)
    FFMOD[np.isnan(FFPHASE)] = np.min(FFPHASE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(FFMOD,  origin='lower', cmap='hsv')
    ax1.set_title('Amplitude [ADC levels]')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    plt.colorbar(im1, ax=ax1, shrink=0.45)
    im2 = ax2.imshow(FFPHASE,  origin='lower', cmap='hsv')
    ax2.set_title('Phase [rad]')
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    plt.colorbar(im2, ax=ax2, shrink=0.45)
    plt.tight_layout()
    #plt.savefig('/Users/breakfastea/Documents/Asmara/images/image.png', dpi=400)
    plt.show()
    
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
        
    # def __getitem2__(self):
    #     for indoor_name, data_indoor in tqdm(self.indoor_scans.items()):
    #         for outdoor_name, data_outdoor in tqdm(self.outdoor_scans.items()):
    #             fusion = 0.14 * data_indoor[1,:,:] + 0.86 * data_outdoor[1,:,:]
    #             np.save(f'/home/lbisognin/ASMARA/data/DATASET/holograms3/{indoor_name}__{outdoor_name}.npy', fusion)
    #             plt.imshow(np.real(fusion) ,origin="lower", cmap="gray")
    #             plt.margins(0)
    #             plt.axis("off")
    #             plt.savefig(f'/home/lbisognin/ASMARA/data/DATASET/images3/{indoor_name}__{outdoor_name}.png', bbox_inches='tight', pad_inches=0)
    #             plt.close()

dataset = LandmineDataset_bis(Path("/home/lbisognin/ASMARA/data/machine_learning_bis/interps/indoor"),
                               Path("/home/lbisognin/ASMARA/data/machine_learning_bis/interps/outdoor"),
                              Path("/home/lbisognin/ASMARA/data/old.processed/meta/mix.csv"))

# for i in tqdm(range(0, 47632)):
#     dataset.__getitem__(i)
# # # #      #355


dataset.__getitem__(47631)


# import shutil
# import os

# # Imposta la directory sorgente e la directory di destinazione
# source_directory = '/home/lbisognin/ASMARA/data/machine_learning_bis/others_refused'
# target_directory = '/home/lbisognin/asmara-main/data/processed/images'

# # Crea la cartella di destinazione se non esiste
# os.makedirs(target_directory, exist_ok=True)

# # Itera attraverso i file nella directory sorgente
# for filename in os.listdir(source_directory):
#     # Estrae il terzo elemento dal nome del file come numero intero
#     class_number = int(filename.split('_')[2])

#     # Controlla se il numero della classe è nell'intervallo desiderato
#         # Costruisce il percorso completo del file sorgente e di destinazione
#     source_path = os.path.join(source_directory, filename)
#     target_path = os.path.join(target_directory, filename)

#     # Sposta il file dalla sorgente alla destinazione
#     shutil.move(source_path, target_path)
#     print(f"File {filename} spostato da {source_directory} a {target_directory}")


# import os
# import random
# import shutil

# def sposta_elementi_casuali(cartella_origine, cartella_destinazione, num_elementi):
#     # Ottieni un elenco di tutti i file nella cartella di origine
#     elenco_file = os.listdir(cartella_origine)
    
#     # Seleziona casualmente num_elementi dall'elenco dei file
#     elementi_da_spostare = random.sample(elenco_file, num_elementi)
    
#     # Sposta i file selezionati nella cartella di destinazione
#     for elemento in elementi_da_spostare:
#         percorso_origine = os.path.join(cartella_origine, elemento)
#         percorso_destinazione = os.path.join(cartella_destinazione, elemento)
#         shutil.move(percorso_origine, percorso_destinazione)

# # Esempio di utilizzo
# cartella_origine = "/home/lbisognin/ASMARA/data/machine_learning_bis/mineclassify_bis/train/other"
# cartella_destinazione = "/home/lbisognin/ASMARA/data/machine_learning_bis/others_refused"
# num_elementi = 8113

# sposta_elementi_casuali(cartella_origine, cartella_destinazione, num_elementi)
