from pathlib import Path

BASEPATH = Path('/home/evivoli/asmara')

datapath = BASEPATH / Path('data')

datarawpath = datapath / Path('raw')
interimpath = datapath / Path('interim')
processedpath = datapath / Path('processed')

# Path for interim data that has been transformed.
hologramspath = datapath / Path('interim/holograms')
imagespath = datapath / Path('interim/images')
inversionspath = datapath / Path('interim/inversions')
metadatapath = datapath / Path('interim/meta')