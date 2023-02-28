# -*- coding: utf-8 -*-
import os
import click
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import random
import numpy as np
import pandas as pd
from PIL import Image
from src.utils.data import if_null_create

from src.utils.spec import *
from src.utils.const import *
from src.utils.struct import Holo, HoloInv


# main get:
# - format, a list of strings [if we want mixed hologram (npy), images (img), inversion (inv) and metadata (meta)]
@click.command()
@click.option('seed', '-s', type=int)
@click.option('indoor_filepath', '-idf', type=click.Path())
@click.option('outdoor_filepath', '-odf', type=click.Path())
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Format of the output files')
@click.option('output_path', '-output', type=click.Path())
def main(seed, indoor_filepath, outdoor_filepath, format, output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    if indoor_filepath is None: indoor_filepath = interimpath
    if outdoor_filepath is None: outdoor_filepath = interimpath
    if output_path is None: output_path = processedpath
    
    splits_path = output_path / Path('splits')

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
    indoor_meta = pd.read_csv(metadatapath / Path('indoor.csv')).T.to_dict().values()
    outdoor_data = pd.read_csv(metadatapath / Path('outdoor.csv')).T.to_dict().values()

    # Set random seed for reproducibility
    random.seed(seed)

    # Set aside 10% of the data for testing
    num_test = int(len(outdoor_data) * 0.1)
    test_meta = random.sample(outdoor_data, num_test)
    train_val_meta = [d for d in outdoor_data if d not in test_meta]

    # Randomly sample 20% of the remaining data for validation
    num_val = int(len(train_val_meta) * 0.2)
    val_meta = random.sample(train_val_meta, num_val)
    train_meta = [d for d in train_val_meta if d not in val_meta]

    mixed_meta = []

    # Combine indoor and outdoor data
    for in_meta in tqdm(indoor_meta):
    
        for out_meta in train_meta + val_meta + test_meta:
            
            # mixing metadata
            meta = {}
            meta['mix_name'] = f'{in_meta["in_file_name"]}__out_{out_meta["out_file_name"]}'
            
            # load holograms ( we need to add at the end "holo.npy")
            in_holo = np.load(hologramspath / Path('indoor') / Path(in_meta['in_file_name']+"_holo.npy"))
            out_holo = np.load(hologramspath / Path('outdoor') / Path(out_meta['out_file_name']+"_holo.npy"))

            mix_holo = Holo(in_holo) + Holo(out_holo)

            if save_npy:
                if_null_create(output_path / Path('holograms'))
                np.save(output_path / Path('holograms') / Path(f'{meta["mix_name"]}.npy'), mix_holo)    

            if save_img:
                if_null_create(output_path / Path('images'))
                mix_img = Image.fromarray((mix_holo * 255).astype(np.uint8))
                mix_img.save(output_path / Path('images') / Path(f'{meta["mix_name"]}.png'))

            if save_inv:
                if_null_create(output_path / Path('inversions'))
                mix_inv = HoloInv(mix_holo)
                np.save(output_path / Path('inversions') / Path(f'{meta["mix_name"]}_inv.npy'), mix_inv)
            
            if save_meta:
                meta = {**meta, **in_meta, **out_meta}
                mixed_meta.append(meta)

    if save_meta:
        if_null_create(output_path / Path('meta'))
        mix_df = pd.DataFrame.from_dict(mixed_meta)
        mix_columns=['mix_name'] + ['in_'+column for column in info['indoor']['columns']] + ['out_'+column for column in info['outdoor']['columns']]
        mix_df.to_csv(output_path / Path('meta') / Path('mix.csv'), index=False, header=True, columns=mix_columns)

    print('[indoor + outdoor] MIX Done!')

    # Shuffle the mixed data
    random.shuffle(mixed_meta)

    # Save the train, val, and test data to CSV files
    header = ['mix_name'] + ['in_'+column for column in info['indoor']['columns']] + ['out_'+column for column in info['outdoor']['columns']]
    train_df = pd.DataFrame(mixed_meta[:-num_val-num_test], columns=header)
    val_df = pd.DataFrame(mixed_meta[-num_val-num_test:-num_test], columns=header)
    test_df = pd.DataFrame(mixed_meta[-num_test:], columns=header)

    # Save the splits
    if_null_create(splits_path)

    train_df.to_csv(splits_path / Path('train.csv'), index=False)
    val_df.to_csv(splits_path / Path( 'val.csv'), index=False)
    test_df.to_csv(splits_path / Path( 'test.csv'), index=False)

    print('[train + val + test] CSV Done!')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
