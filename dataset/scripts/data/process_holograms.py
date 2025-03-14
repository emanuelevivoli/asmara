import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from ..utils.const import *
from ..utils.data import create_annotation
from ..utils.holo import create_inversion
from ..utils.spec import locations, info, params
from ..utils.struct import Holo

logger = logging.getLogger(__name__)

def create_folders(*paths):
    """Creates directories if they don't exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

@click.command()
@click.option('--interpolate', '-i', is_flag=True, help='Interpolate images to 60x60.')
@click.option('--precompute', '-p', is_flag=True, help='Disable MATLAB dependency.')
@click.option('--format', '-f', multiple=False, type=click.Choice(['npy', 'img', 'inv', 'meta']), help='Output formats.')
@click.option('--location', '-l', type=click.Choice(locations), help='Scan location.')
@click.option('--test', '-t', is_flag=True, help='Process only 10 elements.')
def main(interpolate:bool, precompute:bool, format, location, test:bool):
    """Processes raw holographic data and saves results in various formats."""

    logger.info('Starting data processing...')
    
    raw_data_path = Path("../../data/raw_data")
    data_subfolder = "interpolated" if interpolate else "standard"
    base_path = Path(f"../../data/interm_data/{data_subfolder}")

    holograms_path = base_path / "holograms"
    images_path = base_path / "images"
    inversions_path = base_path / "inversions"
    metadata_path = Path("../../data/interm_data/meta")

    precompute = True if interpolate else precompute

    # Initializes MATLAB engine if needed
    if not precompute:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath(os.path.join(BASEPATH, 'matlab')), nargout=0)

    df = pd.read_csv(raw_data_path / "indoor_objects.csv")
    object_info = {row.id: (row.name, row.classification) for _, row in df.iterrows()}

    save_npy, save_img, save_inv, save_meta = (key in format for key in ('npy', 'img', 'inv', 'meta'))
    if not format:
        save_npy = save_img = save_inv = save_meta = True

    locations_to_process = [location] if location else locations

    for loc in tqdm(locations_to_process, desc="Processing locations"):
        metadata = []
        loc_path = raw_data_path / loc

        if not loc_path.exists():
            logger.warning(f"Location {loc} does not exist. Skipping.")
            continue

        names = {f.stem for f in loc_path.iterdir() if f.suffix in {'.log', '.csv'}}
        if test:
            names = list(names)[:10]

        create_folders(holograms_path / loc, images_path / loc, inversions_path / loc, metadata_path)

        for name in tqdm(names, desc=f"Processing {loc} scans"):
            try:
                annotation = create_annotation(name, info, loc, object_info)

                holo_file = holograms_path / loc / f"{name}_holo.npy"

                if not precompute:
                    trace_path = loc_path / f"{name}.csv"
                    log_path = loc_path / f"{name}.log"

                    eng.workspace['trace'] = str(trace_path)
                    eng.workspace['pluto'] = str(log_path)
                    eng.eval("[F, FI, FQ, P_X, P_Y, P_MOD, P_PHASE] = merge_acquisition(trace, pluto);", nargout=0)
                    eng.eval("[MO, PH, H, Hfill] = fast_generate_hologram(F, FI, FQ, 5, P_X, P_Y, P_MOD, P_PHASE, 2, 3);", nargout=0)

                    np_Hfill = np.asarray(eng.workspace['Hfill'], dtype='complex_')
                    if save_npy:
                        np.save(holo_file, np_Hfill)
                else:
                    np_Hfill = np.load(holo_file)

                    if interpolate:
                        np_Hfill = Holo(np_Hfill).interpolate().hologram
                        if save_npy:
                            np.save(holo_file, np_Hfill)

                # Convert hologram to image
                np_Hfill = np.abs(np_Hfill)
                img = Image.fromarray(((np_Hfill - np_Hfill.min()) / (np_Hfill.max() - np_Hfill.min()) * 255).astype(np.uint8))

                if save_img:
                    img.save(images_path / loc / f"{name}.png")

                if save_inv:
                    inversion = create_inversion(
                        images_path / loc / f"{name}.png",
                        MEDIUM_INDEX=params[loc]['MEDIUM_INDEX'],
                        WAVELENGTH=15,
                        SPACING=0.5
                    )
                    np.save(inversions_path / loc / f"{name}_inv.npy", inversion)

                metadata.append(annotation)

            except Exception as e:
                logger.error(f"Error processing {name}: {e}")

        if save_meta:
            df = pd.DataFrame(metadata)
            df.to_csv(metadata_path / f"{loc}.csv", index=False, header=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
