######################
# MIXING IN WITH OUT #
######################

import os
import pandas as pd
from tqdm import tqdm
from src.utils.spec import *
from src.utils.const import *

mix_metadata = []

if not os.path.exists(metapath):
    os.makedirs(metapath)

metadata = {
    'indoor': [],
    'outdoor': [],
}

# LOAD METADATA from csv
metadata['indoor'] = pd.read_csv(metapath / Path('indoor.csv')).T.to_dict().values()
metadata['outdoor'] = pd.read_csv(metapath / Path('indoor.csv')).T.to_dict().values()

for in_meta in tqdm(metadata['indoor']):
    
    for out_meta in metadata['outdoor']:
        
        # mixing metadata
        meta = {}
        meta['mix_name'] = f'{in_meta["in_file_name"]}__out_{out_meta["out_file_name"]}'
        # meta['indoor'] = in_meta
        # meta['outdoor'] = out_meta
        meta = {**meta, **in_meta, **out_meta}
        mix_metadata.append(meta)

mix_df = pd.DataFrame.from_dict(mix_metadata)

mix_columns=['mix_name'] + ['in_'+column for column in info['indoor']['columns']] + ['out_'+column for column in info['outdoor']['columns']]
mix_df.to_csv(metapath / Path('mix.csv'), index=False, header=True, columns=mix_columns)
