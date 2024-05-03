# -*- coding: utf-8 -*-
import os
import click
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd

from PIL import Image
from src.utils.data import if_null_create
from src.utils.holo import create_inversion

from src.utils.spec import *
from src.utils.const import *
from src.utils.struct import Holo

#! MAGIC NUMBER
ALPHA = 0.1399

# main get:
# - format, a list of strings [if we want mixed hologram (npy), images (img), inversion (inv) and metadata (meta)]
@click.command()
@click.option('--interpolate', '-i', is_flag=True, help='If True, we interpolate images and obtain sqared 60x60 images')
@click.option('output_path', '-o', type=click.Path())
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Format of the output files')
def main(interpolate, output_path, format):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    # if output_path is empty, and interpolate is True, we set output_path to inter_inversionspath
    if output_path == None:
        if interpolate:
            output_path = new_processedpath / Path('interps')
        else:
            output_path = new_processedpath / Path('standard')

    if interpolate:
        holograms_path = inter_hologramspath
        inversions_path = inter_inversionspath
    else:
        holograms_path = hologramspath
        inversions_path = inversionspath
    

    # if output_path does not exist, create it
    if_null_create(output_path)
    
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
    
    # Read indoor and outdoor CSV files
    indoor_df = pd.read_csv(metadatapath / Path('indoor.csv'))
    outdoor_df = pd.read_csv(metadatapath / Path('outdoor.csv'))

    # Split indoor and outdoor data
    indoor_meta = indoor_df.T.to_dict().values()
    outdoor_meta = outdoor_df.T.to_dict().values()

    mixed_meta = []

    # Combine indoor and outdoor data
    for in_meta in tqdm(indoor_meta):
    
        for out_meta in outdoor_meta:
            
            # mixing metadata
            meta = {}
            meta['mix_name'] = f'{in_meta["in_file_name"]}__out_{out_meta["out_file_name"]}'
            
            if save_meta:
                meta = {**meta, **in_meta, **out_meta}
                mixed_meta.append(meta)

            # check if at least one of the save_* is True
            if not (save_npy or save_img or save_inv ):
                continue

            # load holograms ( we needxto add at the end "holo.npy")
            in_holo = np.load(holograms_path / Path('indoor') / Path(in_meta['in_file_name']+"_holo.npy"))
            out_holo = np.load(holograms_path / Path('outdoor') / Path(out_meta['out_file_name']+"_holo.npy"))

            # creating Holo objects
            in_holo = Holo(in_holo)
            out_holo = Holo(out_holo)

            mix_holo = Holo(ALPHA*in_holo + (1-ALPHA)*out_holo)

            if save_npy:
                if_null_create(output_path / Path('holograms'))
                mix_holo.save( path = output_path / Path('holograms') / Path(f'{meta["mix_name"]}.npy'))    

            if save_img:
                if_null_create(output_path / Path('images'))
                mix_img = Image.fromarray((mix_holo.hologram * 255).astype(np.uint8)) # ComplexWarning: cast to real discards the imaginary part
                mix_img.save(output_path / Path('images') / Path(f'{meta["mix_name"]}.png'))

            if save_inv:
                if_null_create(output_path / Path('inversions'))    

                in_holo_inv = np.load(inversions_path / Path('indoor') / Path(in_meta['in_file_name']+"_inv.npy"))
                out_holo_inv = np.load(inversions_path / Path('outdoor') / Path(out_meta['out_file_name']+"_inv.npy"))

                # creating Holo objects
                in_holo = Holo(in_holo_inv)
                out_holo = Holo(out_holo_inv)

                mix_inv = Holo(ALPHA*in_holo + (1-ALPHA)*out_holo)
                mix_inv.save(path= output_path / Path('inversions') / Path(f'{meta["mix_name"]}_inv.npy'))
            

    if save_meta:
        if_null_create(output_path / Path('meta'))
        mix_df = pd.DataFrame.from_dict(mixed_meta)
        mix_columns=['mix_name'] + ['in_'+column for column in info['indoor']['columns']] + ['out_'+column for column in info['outdoor']['columns']]
        mix_df.to_csv(output_path / Path('meta') / Path('mix.csv'), index=False, header=True, columns=mix_columns)

    print('[indoor + outdoor] MIX Done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
