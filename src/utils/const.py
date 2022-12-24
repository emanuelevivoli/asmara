from pathlib import Path

BASEPATH = Path('/Users/emanuelevivoli/asmara/')

datapath = BASEPATH / Path('data')

datarawpath = datapath / Path('raw')
metapath = datapath / Path('processed/meta')

# Path for inversions
inversionpath = datapath / Path('interim/inversions')
hologramspath = datapath / Path('interim/holograms')

meta_imagespath = datapath / Path('processed/images_meta')
holo_imagespath = datapath / Path('interim/holo_images')