# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# input/output
import os

# data analysis import
import numpy as np
import pandas as pd

# image processing
from PIL import Image

# utils
from tqdm.auto import tqdm
from src.utils.const import *
from src.utils.data import create_annotation, objects_info
from src.utils.holo import create_inversion
from src.utils.spec import locations, info, params
from src.utils.struct import Holo

#get_numpy
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

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
    
    # if any((abs(P2.MOD_I * np.sin(P2.PHASE_I)) >= 2**11-1) | (abs(P2.MOD_I * np.cos(P2.PHASE_I)) >= 2**11-1)): 
    #     print('WARNING: signal I is saturated.')
    # if any((abs(P2.MOD_Q * np.sin(P2.PHASE_Q)) >= 2**11-1)| (abs(P2.MOD_Q * np.cos(P2.PHASE_Q)) >= 2**11-1)):
    #     print('WARNING: signal Q is saturated.')
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
    px = np.array(P_X[0]) #list index out of range per alcune scansioni qui
    py = np.array(P_Y[0])

    # Definizione della griglia regolare
    xi = np.linspace(min(px), max(px), 62)
    yi = np.linspace(max(py), min(py), 52)

    # Interpolazione dei dati sparsi sulla griglia regolare
    FFMOD = griddata((px, py), P_MOD.values, (xi[None, :], yi[:, None]), method = "linear")
    FFPHASE = griddata((px, py), P_PHASE.values, (xi[None, :], yi[:, None]), method = "linear")
    # zi conterrà i valori interpolati sulla griglia regolare
    FFMOD[np.isnan(FFMOD)] = np.min(FFMOD)
    FFPHASE[np.isnan(FFPHASE)] = np.min(FFPHASE)
    fourier_transform = FFMOD * np.exp(1j * FFPHASE)
        
    return fourier_transform

def create_folder(path, location=None):
    if not os.path.exists(path):
        os.makedirs(path)  

    if location is not None and not os.path.exists(os.path.join(path, location)):
        os.makedirs(os.path.join(path, location))


# main get:
# - precompute, a boolean
# - format, a list of strings
# - location, a string
@click.command()
@click.option('--interpolate', '-i', is_flag=True, help='If True, we interpolate images and obtain sqared 60x60 images')
@click.option('--precompute', '-p', is_flag=True, help='If True, matlab is not needed')
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Format of the output files')
@click.option('--location', '-l', type=click.Choice(locations), help='Location of the scans')
@click.option('--test', '-t', is_flag=True, help='Only test 10 elements')
def main(interpolate, precompute, format, location, test):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if interpolate:
        holograms_path = inter_hologramspath
        images_path = inter_imagespath
        inversions_path = inter_inversionspath
        # additionally, if interpolate, it is also precompute
        precompute = True
    else:
        holograms_path = hologramspath
        images_path = imagespath
        inversions_path = inversionspath

    df, df_dict = objects_info()

    # if format is empty, all are set to True
    if len(format) == 0:
        save_npy = True
        save_img = True
        save_inv = True
        save_meta = True

    else:
        # get save_npy, save_img, save_inv, save_meta from format
        save_npy = 'npy' in format
        save_img = 'img' in format
        save_inv = 'inv' in format
        save_meta = 'meta' in format

    # if location is not specified, process all locations otherwise process only the specified location
    if location is None: locs = locations
    else: locs = [location]

    for location in tqdm(locs):
        
        # -> create HOLOGRAMS
        metadata = []
        
        names = os.listdir( os.path.join(datarawpath, location) )

        # create folder if not exists
        if save_npy: create_folder(holograms_path, location)
        if save_img: create_folder(images_path, location)
        if save_inv: create_folder(inversions_path, location)

        if save_meta: create_folder(metadatapath)

        # instead of doing this:
        #   plutos = [name for name in names if name.endswith('.log')]
        #   sambas = [name for name in names if name.endswith('.csv')]
        # we just take the names without extention:
        names_list = [name.split('.')[0] for name in names]
        union = set(names_list)
        intersection = set([ el for el in union if names_list.count(el) > 1])

        # diff = union - intersection

        columns = info[location]['columns']
        loc_prefix = info[location]['prefix']
        if test:
            intersection = list(intersection)[:10]

        for name in tqdm(intersection):

            pluto = datarawpath / location / f'{name}.log'
            samba = datarawpath / location / f'{name}.csv'

            ######################
            #! CREATE ANNOTATION
            ######################
            
            annotation_obj = create_annotation(name, info, location, df_dict)
            try:
                # if precompute is True, matlab is not needed
                if not precompute:
                    hologram = unifyLog(samba, pluto) #modified
                    if save_npy:
                        # save numpy arrays to file
                        np.save(file=Path(holograms_path) / Path(location) / Path(f'{name}_holo.npy'), arr=hologram)
                else:

                    if interpolate:
                        standard_holograms_path = hologramspath
                        np_Hfill = np.load(file=Path(standard_holograms_path) / Path(location) / Path(f'{name}_holo.npy'))
                        np_Hfill = Holo(np_Hfill).interpolate().hologram
                        if save_npy:
                            np.save(file=Path(holograms_path) / Path(location) / Path(f'{name}_holo.npy'), arr=np_Hfill)
                    else:
                        # load numpy arrays from file
                        np_Hfill = np.load(file=Path(holograms_path) / Path(location) / Path(f'{name}_holo.npy'))
                        

                # convert numpy arrays to image
                
                # np_Hfill = np.abs(np_Hfill)
                # I8 = (((np_Hfill - np_Hfill.min()) / (np_Hfill.max() - np_Hfill.min())) * 255).astype(np.uint8)
                # img = Image.fromarray(I8)

                # if save_img:
                #     # save image to file
                #     img.save(os.path.join(images_path, location, f'{name}.png'))
                
                if save_inv:
                    inversion = create_inversion(os.path.join(images_path, location, f'{name}.png'), MEDIUM_INDEX = params[location]['MEDIUM_INDEX'], WAVELENGTH = 15, SPACING = 0.5 )
                    np.save(file= Path(inversions_path) / Path(location) / Path(f'{name}_inv.npy'), arr=inversion)

                # add info to metadata
                metadata.append(annotation_obj)

            except Exception as e:
                print(e)

        # save metadata
        if save_meta:
            # create pandas dataframe from dicts of lists
            df = pd.DataFrame.from_dict(metadata)
            
            # save pandas to csv
            columns = [f'{loc_prefix}_{column}' for column in columns]
            df.to_csv(Path(metadatapath) / Path(f'{location}.csv'), index=False, header=True, columns=columns)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
