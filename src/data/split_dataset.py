# SPLIT DATASET
# 
# This will create a `data/splits` folder with the splitted dataset. The folder is composed by the following files:
# - `train`: the dataset indexes of the train set
# - `val`: the dataset indexes of the validation set
# - `test`: the dataset indexes of the test set

import os
import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.const import *
from src.utils.data import if_null_create

# main get:
# - format, a list of strings [if we want mixed hologram (npy), images (img), inversion (inv) and metadata (meta)]
@click.command()
@click.option('metadata', '-meta', type=click.Path())
@click.option('output_path', '-output', type=click.Path())
def main(metadata, output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    # Path to the dataset
    if metadata is None: metadata = processedpath / Path('meta') 
    if output_path is None: output_path = processedpath / Path('splits')

    # Read the dataset
    dataset = pd.read_csv(metadata / Path('mix.csv'))

    # Split the dataset (and save the splits indexes)
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    # Save the splits
    if_null_create(output_path)

    train.to_csv(output_path / Path('train.csv'), index=False)
    val.to_csv(output_path / Path( 'val.csv'), index=False)
    test.to_csv(output_path / Path( 'test.csv'), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
