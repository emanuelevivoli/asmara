from pathlib import Path

BASEPATH = Path('/home/breakfastea/projects/asmara')

datapath = BASEPATH / Path('data_bis') 

#datarawpath = datapath / Path('raw')
datarawpath = Path('/home/breakfastea/projects/asmara/data/raw')
interimpath = datapath / Path('interim')
processedpath = datapath / Path('processed')
new_processedpath = datapath / Path('new_processed')

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