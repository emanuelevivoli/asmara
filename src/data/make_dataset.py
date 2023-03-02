# -*- coding: utf-8 -*-
import os
import click
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from src.data.process_holograms import objects_info
from src.utils.data import if_null_create

from src.utils.spec import *
from src.utils.const import *
from src.utils.struct import Holo, HoloInv


# main get:
# - format, a list of strings [if we want mixed hologram (npy), images (img), inversion (inv) and metadata (meta)]
@click.command()
@click.option('--interpolate', '-i', is_flag=True, help='If True, interpolated holograms are used')
@click.option('indoor_filepath', '-idf', type=click.Path())
@click.option('outdoor_filepath', '-odf', type=click.Path())
@click.option('output_path', '-o', type=click.Path())
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Format of the output files')
def main(interpolate, indoor_filepath, outdoor_filepath, output_path, format):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    if indoor_filepath is None: indoor_filepath = interimpath
    if outdoor_filepath is None: outdoor_filepath = interimpath
    if output_path is None: output_path = processedpath

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

            # load holograms ( we need to add at the end "holo.npy")
            in_holo = np.load(hologramspath / Path('indoor') / Path(in_meta['in_file_name']+"_holo.npy"))
            out_holo = np.load(hologramspath / Path('outdoor') / Path(out_meta['out_file_name']+"_holo.npy"))

            mix_holo = Holo(in_holo) + Holo(out_holo)

            if interpolate:
                torch_holo = torch.from_numpy(mix_holo)
                
                # get real and imaginary part of the hologram
                x_real = torch_holo.real.unsqueeze(0).unsqueeze(0)
                x_imag = torch_holo.imag.unsqueeze(0).unsqueeze(0)
                
                # interpolate the hologram
                rescaled_real = F.interpolate(x_real, size=(60, 60), mode='bilinear', align_corners=False)
                rescaled_imag = F.interpolate(x_imag, size=(60, 60), mode='bilinear', align_corners=False)
                
                # fuse real and imaginary part
                torch_holo = torch.complex(rescaled_real, rescaled_imag).squeeze(0).squeeze(0)
                mix_holo = torch_holo.numpy()

            if save_npy:
                if_null_create(output_path / Path('holograms'))
                np.save(output_path / Path('holograms') / Path(f'{meta["mix_name"]}.npy'), mix_holo)    

            if save_img:
                if_null_create(output_path / Path('images'))
                mix_img = Image.fromarray((mix_holo * 255).astype(np.uint8)) # ComplexWarning: cast to real discards the imaginary part
                mix_img.save(output_path / Path('images') / Path(f'{meta["mix_name"]}.png'))

            if save_inv:
                if_null_create(output_path / Path('inversions'))
                mix_inv = HoloInv(mix_holo)
                np.save(output_path / Path('inversions') / Path(f'{meta["mix_name"]}_inv.npy'), mix_inv)
            

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
