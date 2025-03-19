import sys
import os
import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data, holo, spec, struct

logger = logging.getLogger(__name__)

#TODO: check how multiple choices in format and location could be handled safely
@click.command()
@click.option('--interpolate', '-i', is_flag=True, help='Interpolate images to 60x60.')
@click.option('--precompute', '-p', is_flag=True, help='Disable MATLAB dependency.')
@click.option('--format', '-f', multiple=False, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Output formats.')
@click.option('--location', '-l', type=click.Choice(spec.locations), help='Scan location.')
@click.option('--test', '-t', is_flag=True, help='Process only 10 elements.')
@click.option('output_path', '-o', type=click.Path())
def main(interpolate:bool, precompute:bool, format:str, location, test:bool, output_path):
    """Processes raw holographic data and saves results in various formats."""

    logger.info('Starting interm data process ...')

    data_path = Path(__file__).parent.parent.resolve() / "data"
    raw_data_path = data_path / "raw_data"
    interm_data_path = data_path / "interm_data"
    if output_path == None:
        output_path = interm_data_path / ("interpolated" if interpolate else "standard")


    interm_holograms_path = output_path / "holograms"
    interm_images_path = output_path / "images"
    interm_inversions_path = output_path / "inversions"
    interm_metadata_path = interm_data_path / "meta"

    precompute = True if interpolate else precompute

    # Initializes MATLAB engine if needed
    if not precompute:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath(str(Path(__file__).parent.resolve() / 'matlab')), nargout=0)

    df = pd.read_csv(raw_data_path / 'indoor_objects.csv')
    object_info = {row.id: (row.name, row.classification) for _, row in df.iterrows()}

    if not format:
        save_npy = save_img = save_inv = save_meta = True
    else:
        save_npy, save_img, save_inv, save_meta = (key in format for key in ('npy', 'img', 'inv', 'meta'))

    locations_to_process = [location] if location else spec.locations

    for loc in tqdm(locations_to_process, desc="Processing locations"):
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

                metadata.append(data.create_annotation(name, spec.info, loc, object_info))

                if save_img:
                    # Convert hologram to image
                    np_Hfill = np.abs(np_Hfill)
                    img = Image.fromarray(((np_Hfill - np_Hfill.min()) / (np_Hfill.max() - np_Hfill.min()) * 255).astype(np.uint8))
                    data.if_null_create(interm_images_path / loc)
                    img.save(interm_images_path / loc / f"{name}.png")

                if save_inv:
                    inversion = holo.create_inversion(
                        interm_images_path / loc / f"{name}.png",
                        MEDIUM_INDEX=spec.params[loc]['MEDIUM_INDEX'],
                        WAVELENGTH=15,
                        SPACING=0.5
                    )
                    data.if_null_create(interm_inversions_path / loc)
                    np.save(interm_inversions_path / loc / f"{name}_inv.npy", inversion)

            except Exception as e:
                logger.error(f"Error processing {name}: {e}")

        if save_meta:
            print('DHN')
            df = pd.DataFrame(metadata)
            data.if_null_create(interm_metadata_path)
            df.to_csv(interm_metadata_path / f"{loc}.csv", index=False, header=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
