# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data, spec, struct

#! MAGIC NUMBER
ALPHA = 0.143

logger = logging.getLogger(__name__)

# main get:
# - format, a list of strings [if we want mixed hologram (npy), images (img), inversion (inv) and metadata (meta)]
#TODO: check how multiple choices in format and location could be handled safely
@click.command()
@click.option('--interpolate', '-i', is_flag=True, help='If True, we interpolate images and obtain sqared 60x60 images')
@click.option('output_path', '-o', type=click.Path())
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Format of the output files')
def main(interpolate, output_path, format):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('Starting final data process ...')

    # if output_path is empty, and interpolate is True, we set output_path to inter_inversionspath
    data_path = Path(__file__).parent.parent.resolve() / "data"
    interm_data_path = data_path/ "interm_data"
    processed_data_path = data_path / "processed_data"
    if output_path == None:
        output_path = processed_data_path / ("interpolated" if interpolate else "standard")

    processed_holograms_path = output_path / "holograms"
    processed_images_path = output_path / "images"
    processed_inversions_path = output_path / "inversions"
    processed_metadata_path = output_path / "meta"

    interm_metadata_path = interm_data_path / "meta"
    interm_holograms_path = interm_data_path / ("interpolated" if interpolate else "standard") / "holograms"
    interm_inversions_path = interm_data_path / ("interpolated" if interpolate else "standard") / 'inversions'

    # if output_path does not exist, create it
    data.if_null_create(output_path)

    # if format is empty, all are set to True
    if not format:
        save_npy = save_img = save_inv = save_meta = True
    else:
        save_npy, save_img, save_inv, save_meta = (key in format for key in ('npy', 'img', 'inv', 'meta'))

    # Read indoor and outdoor CSV files
    indoor_df = pd.read_csv(interm_metadata_path / 'indoor.csv')
    outdoor_df = pd.read_csv(interm_metadata_path / 'outdoor.csv')

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
            in_holo = np.load(interm_holograms_path / 'indoor' / (in_meta['in_file_name']+"_holo.npy"))
            out_holo = np.load(interm_holograms_path / 'outdoor' / (out_meta['out_file_name']+"_holo.npy"))

            # creating Holo objects
            in_holo = struct.Holo(in_holo)
            out_holo = struct.Holo(out_holo)

            mix_holo = struct.Holo(ALPHA*in_holo + (1-ALPHA)*out_holo)

            if save_npy:
                data.if_null_create(processed_holograms_path)
                mix_holo.save( processed_holograms_path / f'{meta["mix_name"]}.npy')

            if save_img:
                data.if_null_create(processed_images_path)
                mix_holo = np.abs(mix_holo.hologram)
                mix_img = Image.fromarray((((mix_holo - mix_holo.min()) / (mix_holo.max() - mix_holo.min())) * 255).astype(np.uint8))  # ComplexWarning: cast to real discards the imaginary part
                mix_img.save(processed_images_path / f'{meta["mix_name"]}.png')

            if save_inv:
                data.if_null_create(processed_inversions_path)

                in_holo_inv = np.load(interm_inversions_path / 'indoor' / (in_meta['in_file_name']+"_inv.npy"))
                out_holo_inv = np.load(interm_inversions_path / 'outdoor' / (out_meta['out_file_name']+"_inv.npy"))

                # creating Holo objects
                in_holo = struct.Holo(in_holo_inv)
                out_holo = struct.Holo(out_holo_inv)

                mix_inv = struct.Holo(ALPHA*in_holo + (1-ALPHA)*out_holo)
                mix_inv.save(processed_inversions_path / f'{meta["mix_name"]}_inv.npy')

    if save_meta:
        data.if_null_create(processed_metadata_path)
        mix_df = pd.DataFrame.from_dict(mixed_meta)
        mix_columns=['mix_name'] + ['in_'+column for column in spec.info['indoor']['columns']] + ['out_'+column for column in spec.info['outdoor']['columns']]
        mix_df.to_csv(processed_metadata_path / 'mix.csv', index=False, header=True, columns=mix_columns)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
