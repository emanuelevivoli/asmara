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

# hologram processing
import holopy as hp

# image processing
from PIL import Image

# utils
from tqdm.auto import tqdm
from src.utils.const import *
from src.utils.spec import locations, info, params

def matlab_settings():
    # matlab imports
    import matlab.engine
    # create a global variable
    global eng
    eng = matlab.engine.start_matlab()
    s = eng.genpath(os.path.join(BASEPATH,'matlab'))
    eng.addpath(s, nargout=0)

def objects_info(filename='indoor_objects.csv', key='id', columns=['name', 'classification']):
    assert len(columns) >= 2
    df = pd.read_csv(Path(datarawpath) / Path(filename))
    df_dict = {k: (v1, v2) for k, (v1, v2) in zip(
        df[key], zip(df[columns[0]], df[columns[1]]))}
    return df, df_dict

def create_annotation(name, info, location, df_dict):
    
    indexes = info[location]['indexes']
    keys = info[location]['keys']
    prefix = info[location]['prefix']
    
    name_list = name.split('_')
    
    obj = {}
    for c_index, c_key in zip(indexes, keys):
        obj[f'{prefix}_{c_key}'] = name_list[c_index]

    if len(name_list) > len(indexes):
        if location == 'indoor':
            inclination = 20
        else:
            additional = name_list[3]
    else:
        if location == 'indoor':
            inclination = 0
        else:
            additional = None

    obj[f'{prefix}_location'] = location        
    obj[f'{prefix}_file_name'] = name

    if location == 'indoor':
        obj[f'{prefix}_distance_from_source'] = 8 if obj[f'{prefix}_distance_from_source'] == 'low' else 4 if obj[f'{prefix}_distance_from_source'] == 'bas' else None
        obj[f'{prefix}_inclination'] = inclination
        category_, name_ = df_dict.get(int(obj[f'{prefix}_id']), (None, None))
        
        # name = "pmn-4"
        obj[f'{prefix}_name'] = name_
    else:
        category_ = 'ground-smarta'
        obj[f'{prefix}_additional'] = additional
    
    # category = "mine"
    obj[f'{prefix}_category'] = category_

    return obj


def create_inversion(img,
                    MEDIUM_INDEX = None,
                    WAVELENGTH = 15,
                    SPACING = 0.5 ):
    
    assert MEDIUM_INDEX is not None, "MEDIUM_INDEX is not defined"
    
    # load image from file
    raw_holo = hp.load_image(img, 
                            medium_index=MEDIUM_INDEX, 
                            illum_wavelen=WAVELENGTH,
                            illum_polarization=(1,0), 
                            spacing=SPACING)
                        
    zstack = np.linspace(0, 30, 61)
    rec_vol = hp.propagate(raw_holo, zstack)

    return rec_vol

def create_folder(path, location):
    if not os.path.exists(path):
        os.makedirs(path)  

    if not os.path.exists(os.path.join(path, location)):
        os.makedirs(os.path.join(path, location))


# main get:
# - precompute, a boolean
# - format, a list of strings
# - location, a string
@click.command()
@click.option('--precompute', '-p', is_flag=True, help='If True, matlab is not needed')
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Format of the output files')
@click.option('--location', '-l', type=click.Choice(locations), help='Location of the scans')
def main(precompute, format, location):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # if precompute is True, matlab is not needed
    if not precompute:
        matlab_settings()

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
        if save_npy: create_folder(hologramspath, location)
        if save_img: create_folder(imagespath, location)
        if save_inv: create_folder(inversionspath, location)
        if save_meta: create_folder(metadatapath, location)

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

        for name in tqdm(intersection):

            pluto = f'{name}.log'
            samba = f'{name}.csv'

            ######################
            #! CREATE ANNOTATION
            ######################

            annotation_obj = create_annotation(name, info, location, df_dict)

            try:
                # if precompute is True, matlab is not needed
                if not precompute:
                    eng.workspace['pluto']= os.path.join(datarawpath, location, pluto)
                    eng.workspace['trace']= os.path.join(datarawpath, location, samba)

                    eng.eval(f"[F,FI,FQ,P_X,P_Y,P_MOD,P_PHASE] = merge_acquisition(trace, pluto);",nargout=0)
                    eng.eval(f"[MO, PH, H, Hfill] = fast_generate_hologram(F, FI, FQ, 5, P_X, P_Y, P_MOD, P_PHASE, 2, 3);",nargout=0)

                    # get metlab matrixes as numpy arrays
                    Hfill = eng.workspace['Hfill']
                    np_Hfill = np.asarray(Hfill, dtype = 'complex_')

                    if save_npy:
                        # save numpy arrays to file
                        np.save(file=Path(hologramspath) / Path(location) / Path(f'{name}_holo.npy'), arr=np_Hfill)
                else:
                    # load numpy arrays from file
                    np_Hfill = np.load(file=Path(hologramspath) / Path(location) / Path(f'{name}_holo.npy'))

                # convert numpy arrays to image
                np_Hfill = np.abs(np_Hfill)
                I8 = (((np_Hfill - np_Hfill.min()) / (np_Hfill.max() - np_Hfill.min())) * 255).astype(np.uint8)
                img = Image.fromarray(I8)

                if save_img:
                    # save image to file
                    img.save(os.path.join(imagespath, location, f'{name}.png'))
                
                if save_inv:
                    inversion = create_inversion(os.path.join(imagespath, location, f'{name}.png'), MEDIUM_INDEX = params[location]['MEDIUM_INDEX'], WAVELENGTH = 15, SPACING = 0.5 )
                    np.save(file= Path(inversionspath) / Path(location) / Path(f'{name}_inv.npy'), arr=inversion)

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
