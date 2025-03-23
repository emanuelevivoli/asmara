import sys
import os
from typing import Optional
import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import holo, struct, data
from utils.info import info

params = {
    'indoor':{
        'MEDIUM_INDEX': 1,
    },
    'outdoor':{
        'MEDIUM_INDEX': 4,
    }
}

logger = logging.getLogger(__name__)

#FIX: inclination and additional are possibly unbound
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
            raise ValueError(f'Additional info not found for {name}')
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
        obj[f'{prefix}_name'] = name_
    else:
        category_ = 'ground-smarta'
        obj[f'{prefix}_additional'] = additional

    obj[f'{prefix}_category'] = category_

    return obj

def make_interm(interpolate:bool = False, precompute:bool = False, format:tuple[str, ...] = ('npy', 'img', 'inv', 'meta'), locations:tuple[str, ...] = ('outdoor', 'indoor'), test:bool = False, output_path:Optional[Path] = None):
    """Processes raw holographic data into their interm version"""

    logger.info('Starting interm data process ...')

    data_path = Path(__file__).parent.parent.resolve() / "data"
    raw_data_path = data_path / "raw_data"
    interm_data_path = data_path / "interm_data"
    if not output_path:
        output_path = interm_data_path / ("interpolated" if interpolate else "standard")
        interm_metadata_path = output_path.parent / "meta"
    else:
        interm_data_path = output_path / "meta"

    output_path = Path(output_path)
    interm_holograms_path = output_path / "holograms"
    interm_images_path = output_path / "images"
    interm_inversions_path = output_path / "inversions"

    precompute = True if interpolate else precompute

    if not precompute:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath(str(Path(__file__).parent.resolve() / 'matlab')), nargout=0)

    df = pd.read_csv(raw_data_path / 'indoor_objects.csv')
    object_info = {row.id: (row.name, row.classification) for _, row in df.iterrows()}

    save_npy, save_img, save_inv, save_meta = (key in format for key in ('npy', 'img', 'inv', 'meta'))

    for loc in tqdm(locations, desc="Processing locations"):

        metadata = []
        loc_path = raw_data_path / loc

        if not loc_path.exists():
            logger.warning(f"Location {loc} does not exist. Skipping.")
            continue

        names = {f.stem for f in loc_path.iterdir() if f.suffix in {'.log', '.csv'}}
        if test:
            names = list(names)[:10]

        for name in tqdm(names, desc=f"Processing {loc} scans"):
            try:

                holo_file = interm_data_path / "standard/holograms" / loc / f"{name}_holo.npy"

                if not precompute:
                    trace_path = loc_path / f"{name}.csv"
                    log_path = loc_path / f"{name}.log"

                    eng.workspace['trace'] = str(trace_path.resolve())
                    eng.workspace['pluto'] = str(log_path.resolve())
                    eng.eval("[F, FI, FQ, P_X, P_Y, P_MOD, P_PHASE] = merge_acquisition(trace, pluto);", nargout=0)
                    eng.eval("[MO, PH, H, Hfill] = fast_generate_hologram(F, FI, FQ, 5);", nargout=0)

                    np_Hfill = np.asarray(eng.workspace['Hfill'], dtype=np.complex128)

                    if save_npy:
                        data.if_null_create(interm_holograms_path / loc)
                        np.save(holo_file, np_Hfill)
                else:
                    np_Hfill = np.load(holo_file)

                    if interpolate:
                        np_Hfill = struct.Holo(np_Hfill).interpolate().hologram
                        if save_npy:
                            data.if_null_create(interm_holograms_path / loc)
                            np.save(interm_holograms_path / loc / f"{name}_holo.npy", np_Hfill)

                metadata.append(create_annotation(name, info, loc, object_info))

                if save_img:
                    # Convert hologram to image
                    np_Hfill = np.abs(np_Hfill)
                    img = Image.fromarray(((np_Hfill - np_Hfill.min()) / (np_Hfill.max() - np_Hfill.min()) * 255).astype(np.uint8))
                    data.if_null_create(interm_images_path / loc)
                    img.save(interm_images_path / loc / f"{name}.png")

                if save_inv:
                    inversion = holo.create_inversion(
                        interm_images_path / loc / f"{name}.png",
                        MEDIUM_INDEX=params[loc]['MEDIUM_INDEX'],
                        WAVELENGTH=15,
                        SPACING=0.5
                    )
                    data.if_null_create(interm_inversions_path / loc)
                    np.save(interm_inversions_path / loc / f"{name}_inv.npy", inversion)

            except Exception as e:
                logger.error(f"Error processing {name}: {e}")

        if save_meta:
            df = pd.DataFrame(metadata)
            data.if_null_create(interm_metadata_path)
            df.to_csv(interm_metadata_path / f"{loc}.csv", index=False, header=True)

@click.command()
@click.option('--interpolate', '-i', is_flag=True, default=False, 
              help="Resize images to 60x60 resolution. This automatically enables '--precompute'. Run this command once without interpolation first to generate the required base files.")
@click.option('--precompute', '-p', is_flag=True, default=False, 
              help="Bypass MATLAB dependency by using precomputed files. Ensure you have previously run the function at least once to generate non-interpolated '.npy' files.")
@click.option('--format', '-f', multiple=True, type=click.Choice(['npy', 'img', 'inv', 'meta']), 
              default=('npy', 'img', 'inv', 'meta'), 
              help="Choose output formats: 'npy' (NumPy arrays), 'img' (images), 'inv' (inverse data), 'meta' (metadata). Defaults to all formats. Repeat the flag for multiple choices.")
@click.option('--locations', '-l', multiple=True, type=click.Choice(['outdoor', 'indoor']), 
              default=('outdoor', 'indoor'), 
              help="Specify scan locations: 'outdoor' or 'indoor'. Defaults to both. Repeat the flag for multiple choices.")
@click.option('--test', '-t', is_flag=True, default=False, 
              help="Enable test mode to process only 10 elements for quick validation.")
@click.option('--output_path', '-o', type=click.Path(resolve_path=True), default=None,
              help="Set the base output directory. The folder structure created will match the default one")

def cli_make_interm(interpolate:bool = False, precompute:bool = False, format:tuple[str, ...] = ('npy', 'img', 'inv', 'meta'), locations:tuple[str, ...] = ('outdoor', 'indoor'), test:bool = False, output_path:Optional[Path] = None):
    make_interm(interpolate, precompute, format, locations, test, output_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli_make_interm()
