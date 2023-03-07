from pathlib import Path

BASEPATH = Path('/Users/emanuelevivoli/Projects/asmara')

datapath = BASEPATH / Path('data')

datarawpath = datapath / Path('raw')
interimpath = datapath / Path('interim')
processedpath = datapath / Path('processed')

# Path for interim data that has been transformed.
metadatapath = datapath / Path('interim/meta')

# standard
hologramspath = datapath / Path('interim/standard/holograms')
imagespath = datapath / Path('interim/standard/images')
inversionspath = datapath / Path('interim/standard/inversions')

# interpolated
inter_hologramspath = datapath / Path('interim/interps/holograms')
inter_imagespath = datapath / Path('interim/interps/images')
inter_inversionspath = datapath / Path('interim/interps/inversions')