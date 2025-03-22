import os
import sys
from typing import Optional
import click
import logging
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data, struct
from utils.info import info

#! MAGIC NUMBER FROM HOLOMINE PAPER
ALPHA = 0.143

logger = logging.getLogger(__name__)

def make_processed(interpolate:bool = False, output_path:Optional[Path] = None, format:tuple[str, ...] = ('npy', 'img', 'inv', 'meta')):
    """Processes interm holographic data into their processed version"""

    logger.info('Starting final data process ...')

    data_path = Path(__file__).parent.parent.resolve() / "data"
    interm_data_path = data_path/ "interm_data"
    processed_data_path = data_path / "processed_data"
    if not output_path:
        output_path = processed_data_path / ("interpolated" if interpolate else "standard")

    output_path = Path(output_path)
    processed_holograms_path = output_path / "holograms"
    processed_images_path = output_path / "images"
    processed_inversions_path = output_path / "inversions"
    processed_metadata_path = output_path / "meta"

    interm_metadata_path = interm_data_path / "meta"
    interm_holograms_path = interm_data_path / ("interpolated" if interpolate else "standard") / "holograms"
    interm_inversions_path = interm_data_path / ("interpolated" if interpolate else "standard") / 'inversions'

    data.if_null_create(output_path)

    if not format:
        save_npy = save_img = save_inv = save_meta = True
    else:
        save_npy, save_img, save_inv, save_meta = (key in format for key in ('npy', 'img', 'inv', 'meta'))

    indoor_df = pd.read_csv(interm_metadata_path / 'indoor.csv')
    outdoor_df = pd.read_csv(interm_metadata_path / 'outdoor.csv')

    indoor_meta = indoor_df.T.to_dict().values()
    outdoor_meta = outdoor_df.T.to_dict().values()

    mixed_meta = []

    for in_meta in tqdm(indoor_meta):

        for out_meta in outdoor_meta:

            # mixing metadata
            meta = {}
            meta['mix_name'] = f'{in_meta["in_file_name"]}__out_{out_meta["out_file_name"]}'

            if save_meta:
                meta = {**meta, **in_meta, **out_meta}
                mixed_meta.append(meta)

            if not (save_npy or save_img or save_inv ):
                continue

            in_holo = np.load(interm_holograms_path / 'indoor' / (in_meta['in_file_name']+"_holo.npy"))
            out_holo = np.load(interm_holograms_path / 'outdoor' / (out_meta['out_file_name']+"_holo.npy"))

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

                in_holo = struct.Holo(in_holo_inv)
                out_holo = struct.Holo(out_holo_inv)

                mix_inv = struct.Holo(ALPHA*in_holo + (1-ALPHA)*out_holo)
                mix_inv.save(processed_inversions_path / f'{meta["mix_name"]}_inv.npy')

    if save_meta:
        data.if_null_create(processed_metadata_path)
        mix_df = pd.DataFrame.from_dict(mixed_meta)
        mix_columns=['mix_name'] + ['in_'+column for column in info['indoor']['columns']] + ['out_'+column for column in info['outdoor']['columns']]
        mix_df.to_csv(processed_metadata_path / 'mix.csv', index=False, header=True, columns=mix_columns)

@click.command()
@click.option('--interpolate', '-i', is_flag=True, default=False, 
              help="Resize images to 60x60 resolution. This automatically enables '--precompute'. Run this command once without interpolation first to generate the required base files.")
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), 
              default=('npy', 'img', 'inv', 'meta'), 
              help="Choose output formats: 'npy' (NumPy arrays), 'img' (images), 'inv' (inverse data), 'meta' (metadata). Defaults to all formats. Repeat the flag for multiple choices.")
@click.option('--output_path', '-o', type=click.Path(resolve_path=True), default=None,
              help="Set the base output directory. The folder structure created will match the default one")

def cli_make_processed(interpolate:bool = False, output_path:Optional[Path] = None, format:tuple[str, ...] = ('npy', 'img', 'inv', 'meta')):
    make_processed(interpolate, output_path, format)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli_make_processed()
