# -*- coding: utf-8 -*-
import os
import itertools
import click
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import random
import numpy as np
import pandas as pd
from PIL import Image
from src.data.process_holograms import objects_info
from src.utils.data import if_null_create

from src.utils.spec import *
from src.utils.const import *


# main get:
# - format, a list of strings [if we want mixed hologram (npy), images (img), inversion (inv) and metadata (meta)]
@click.command()
@click.option('--seed', '-s', type=int)
@click.option('--out', '-o', type=click.Path())
def main(seed, out):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    splits_path = out
    if splits_path is None: splits_path = processedpath / Path('splits') / Path(f'{seed}')

    # if splits_path does not exist, create it
    if_null_create(splits_path)
    
    # Read indoor and outdoor CSV files
    indoor_df = pd.read_csv(metadatapath / Path('indoor.csv'))
    outdoor_df = pd.read_csv(metadatapath / Path('outdoor.csv'))

    objects, _ = objects_info()
    
    # 'binary': 'classification', # 'mine' or 'not mine'
    # 'multi': 'classification', # 'mine', 'clutter' or 'archaelogical'
    # 'multi_id': 'name', # 'pmn1', 'pmn2', ...

    # Split indoor and outdoor data
    indoor_meta = indoor_df.T.to_dict().values()
    outdoor_meta = outdoor_df.T.to_dict().values()

    # Set random seed for reproducibility
    random.seed(seed)

    # for the seed, we create three different splits: 
    # - binary split (where mines are 1 and other is 0)
    # - multi-class split (where mines are 1, clutter is 0 and archaelogical is 2)
    # - multi-class id split (where the classes are from 0 to 13)
    labels = {
        'bin': {
            'mine': 1,
            'clutter': 0,
            'archeology': 0
        },
        'tri': {
            'mine': 1,
            'clutter': 0,
            'archeology': 2
        }
    }
    
    # Get the ids of the objects
    X = np.array(objects['id'].values.tolist())
    
    #! Binary split
    by = [labels['bin'][obj] for obj in objects['classification'].values.tolist()]
    bX_train, bX_test, by_train, by_test = train_test_split(X, by, stratify=by, test_size=0.2, random_state=seed)
    bX_train, bX_val, by_train, by_val = train_test_split(bX_train, by_train, stratify=by_train, test_size=0.2, random_state=seed)
    print(bX_train, bX_val, bX_test)

    #! Trinary split
    ty = np.array([labels['tri'][obj] for obj in objects['classification'].values.tolist()])
    mine_idx = [0, 1, 2, 3, 4, 5, 6]
    clut_idx = [7, 8, 9]
    arch_idx = [10, 11, 12]
    
    random.shuffle(mine_idx)
    random.shuffle(clut_idx)
    random.shuffle(arch_idx)

    tX_val = X[np.array([mine_idx[0], clut_idx[0], arch_idx[0]])]
    ty_val = ty[np.array([mine_idx[0], clut_idx[0], arch_idx[0]])]
    tX_test = X[np.array([mine_idx[1], clut_idx[1], arch_idx[1]])]
    ty_test = ty[np.array([mine_idx[1], clut_idx[1], arch_idx[1]])]
    tX_train = X[np.array(mine_idx[2:] + clut_idx[2:] + arch_idx[2:])]
    ty_train = ty[np.array(mine_idx[2:] + clut_idx[2:] + arch_idx[2:])]

    print(tX_train, tX_val, tX_test)
    
    #! Multi-class id split
    # the X became the combination of views which is 
    # -------
    orientation = ['A', 'B', 'C', 'D']
    inclination = [20, 0]
    hight = [8, 4]
    # ----

    mX = [f'{x[0]}_{x[1]}_{x[2]}' for x in itertools.product(orientation, hight, inclination)]
    my = [x[1] for x in itertools.product(orientation, hight, inclination)]
    mX_train, mX_test, my_train, my_test = train_test_split(mX, my, stratify=my, test_size=0.2, random_state=seed)
    mX_train, mX_val, my_train, my_val = train_test_split(mX_train, my_train, stratify=my_train, test_size=0.2, random_state=seed)
    
    print(mX_train, mX_val, mX_test)

    # todo
    # now we have to create the metadata for each split, starting from the mix.csv file
    bin_split = {
        'train': [],
        'val': [],
        'test': []
    }
    tri_split = {
        'train': [],
        'val': [],
        'test': []
    }
    multi_split = {
        'train': [],
        'val': [],
        'test': []
    }

    b_train = False
    b_val = False
    

    t_train = False
    t_val = False
    

    m_train = False
    m_val = False
    
    
    # Combine indoor and outdoor data
    for in_meta in tqdm(indoor_meta):
        
        # check for the in_meta['in_id'] 
        # mix_name,in_file_name,in_id,in_category,in_name,in_orientation,in_distance_from_source,in_inclination,in_location,out_file_name,out_id,out_category,out_orientation,out_location,out_additional
        # in_low_08_A__out_51_25,in_low_08_A,8,wood-cylinder,clutter,A,8,0,indoor,51_25,51,ground-smarta,25,outdoor,

        # binary split
        if int(in_meta['in_id']) in bX_train:
            b_train = True
        elif int(in_meta['in_id']) in bX_val:
            b_val = True

        # trinary split
        if int(in_meta['in_id']) in tX_train:
            t_train = True
        elif int(in_meta['in_id']) in tX_val:
            t_val = True

        # multi-class split
        if f"{in_meta['in_orientation']}_{in_meta['in_distance_from_source']}_{in_meta['in_inclination']}" in mX_train:
            m_train = True
        elif f"{in_meta['in_orientation']}_{in_meta['in_distance_from_source']}_{in_meta['in_inclination']}" in mX_val:
            m_val = True

        for out_meta in outdoor_meta:
            
            # mixing metadata
            meta = {}
            meta['mix_name'] = f"{in_meta['in_file_name']}__out_{out_meta['out_file_name']}"
            meta = {**meta, **in_meta, **out_meta}
            
            # binary split
            if b_train:
                bin_split['train'].append(meta)
            elif b_val:
                bin_split['val'].append(meta)
            else:
                bin_split['test'].append(meta)

            # trinary split
            if t_train:
                tri_split['train'].append(meta)
            elif t_val:
                tri_split['val'].append(meta)
            else:
                tri_split['test'].append(meta)

            # multi-class split
            if m_train:
                multi_split['train'].append(meta)
            elif m_val:
                multi_split['val'].append(meta)
            else:
                multi_split['test'].append(meta)

        b_train = False
        b_val = False
        

        t_train = False
        t_val = False
        

        m_train = False
        m_val = False


    mix_columns=['mix_name'] + ['in_'+column for column in info['indoor']['columns']] + ['out_'+column for column in info['outdoor']['columns']]
    
    if_null_create(splits_path / Path('binary'))
    if_null_create(splits_path / Path('trinary'))
    if_null_create(splits_path / Path('multi'))

    # Save metadata
    # -------------

    # Binary split
    train_df = pd.DataFrame.from_dict(bin_split['train'])
    val_df = pd.DataFrame.from_dict(bin_split['val'])
    test_df = pd.DataFrame.from_dict(bin_split['test'])

    
    train_df.to_csv(splits_path / Path('binary') / Path('train.csv'), index=False, header=True, columns=mix_columns)
    val_df.to_csv(splits_path / Path('binary') / Path('val.csv'), index=False, header=True, columns=mix_columns)
    test_df.to_csv(splits_path / Path('binary') / Path('test.csv'), index=False, header=True, columns=mix_columns)

    # Trinary split
    # -------------
    train_df = pd.DataFrame.from_dict(tri_split['train'])
    val_df = pd.DataFrame.from_dict(tri_split['val'])
    test_df = pd.DataFrame.from_dict(tri_split['test'])

    train_df.to_csv(splits_path / Path('trinary') / Path('train.csv'), index=False, header=True, columns=mix_columns)
    val_df.to_csv(splits_path / Path('trinary') / Path('val.csv'), index=False, header=True, columns=mix_columns)
    test_df.to_csv(splits_path / Path('trinary') / Path('test.csv'), index=False, header=True, columns=mix_columns)

    # Multi-class split
    # -----------------
    train_df = pd.DataFrame.from_dict(multi_split['train'])
    val_df = pd.DataFrame.from_dict(multi_split['val'])
    test_df = pd.DataFrame.from_dict(multi_split['test'])

    train_df.to_csv(splits_path / Path('multi') / Path('train.csv'), index=False, header=True, columns=mix_columns)
    val_df.to_csv(splits_path / Path('multi') / Path('val.csv'), index=False, header=True, columns=mix_columns)
    test_df.to_csv(splits_path / Path('multi') / Path('test.csv'), index=False, header=True, columns=mix_columns)

    print('[indoor + outdoor] Splits Done!')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
