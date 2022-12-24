import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from src.utils.const import *
from src.utils.spec import locations, prefixes, positions, info

import matlab.engine
eng = matlab.engine.start_matlab()
s = eng.genpath(os.path.join(BASEPATH, 'matlab'))
eng.addpath(s, nargout=0)

df = pd.read_csv(datapath / Path('raw/indoor_objects.csv'))
df_dict = {k: (v1, v2) for k, (v1, v2) in zip(
    df['id'], zip(df['name'], df['classification']))}

###############
# SPECIFICS
###############

metadata = {
    'indoor': [],
    'outdoor': [],
}

def create_annotation(name, info, location):
    
    indexes = info[location]['indexes']
    keys = info[location]['keys']
    prefix = info[location]['prefix']
    
    name_list = name.split('_')
    if len(name_list) > len(indexes):
        custom_indexes = np.ones(len(indexes), dtype=int)
        custom_indexes[0] = 0
        custom_indexes[1] = 0
        custom_indexes[2] = 0
        custom_indexes += indexes
        if location == 'indoor':
            inclination = 20
        else:
            additional = name_list[3]
    else:
        if location == 'indoor':
            inclination = 0
        else:
            additional = None
        custom_indexes = indexes

    obj = {}
    for c_index, c_key in zip(custom_indexes, keys):
        obj[f'{prefix}_{c_key}'] = name_list[c_index]

    obj[f'{prefix}_location'] = location        
    obj[f'{prefix}_file_name'] = name

    if location == 'indoor':
        obj[f'{prefix}_distance_from_source'] = 8 if obj[f'{prefix}_distance_from_source'] == 'low' else 4 if obj[f'{prefix}_distance_from_source'] == 'bas' else None
        obj[f'{prefix}_inclination'] = inclination
        category_, name_ = df_dict.get(int(obj[f'{prefix}_id']), (None, None))
        
        # name = "pmn-4"
        obj[f'{prefix}_name'] = name_
    else:
        category_ = 'ground-smarta'
        obj[f'{prefix}_additional'] = additional
    
    # category = "mine"
    obj[f'{prefix}_category'] = category_

    return obj

####################
# create HOLOGRAMS #
####################

for location in tqdm(locations):
    
    names = os.listdir( os.path.join(datarawpath, location) )
    
    # create folder if not exists
    if not os.path.exists(hologramspath):
        os.makedirs(hologramspath)  

    if not os.path.exists(os.path.join(hologramspath, location)):
        os.makedirs(os.path.join(hologramspath, location))  

    # instead of doing this:
    #   plutos = [name for name in names if name.endswith('.log')]
    #   sambas = [name for name in names if name.endswith('.csv')]
    # we just take the names without extention:
    names_list = [name.split('.')[0] for name in names]
    union = set(names_list)
    intersection = set([ el for el in union if names_list.count(el) > 1])

    diff = union - intersection

    columns = info[location]['columns']
    loc_prefix = info[location]['prefix']

    for name in tqdm(intersection):

        pluto = f'{name}.log'
        samba = f'{name}.csv'

        ######################
        #! CREATE ANNOTATION
        ######################

        annotation_obj = create_annotation(name, info, location)

        for prefix in prefixes[location]:
            try:
                eng.workspace['pluto']= os.path.join(datarawpath, location, pluto)
                eng.workspace['trace']= os.path.join(datarawpath, location, samba)

                eng.eval(f"[F,FI,FQ,P_X,P_Y,P_MOD,P_PHASE] = merge_acquisition(trace, pluto);",nargout=0)
                eng.eval(f"[MO, PH, H, Hfill] = fast_generate_hologram(F, FI, FQ, 5, P_X, P_Y, P_MOD, P_PHASE, 2, 3);",nargout=0)

                #!################
                #! VARIABLES
                #!################
                # MO = eng.workspace['MO']
                # np_MO = np.asarray(MO, dtype = 'complex_')

                # PH = eng.workspace['PH']
                # np_PH = np.asarray(PH, dtype = 'complex_')

                # H = eng.workspace['H']
                # np_H = np.asarray(H, dtype = 'complex_')
                
                Hfill = eng.workspace['Hfill']
                np_Hfill = np.asarray(Hfill, dtype = 'complex_')
                np.save(file=os.path.join(hologramspath, location, f'{name}.npy'), arr=np_Hfill)

                # np.save(file=os.path.join(hologramspath, location, f'{name}_MO.npy'), arr=np_MO)
                # np.save(file=os.path.join(hologramspath, location, f'{name}_PH.npy'), arr=np_PH)
                # np.save(file=os.path.join(hologramspath, location, f'{name}_H.npy'), arr=np_H)
                # np.save(file=os.path.join(hologramspath, location, f'{name}_Hfill.npy'), arr=np_Hfill)

                metadata[location].append(annotation_obj)

            except Exception as e:
                print(e)

    df = pd.DataFrame.from_dict(metadata[location])
    
    columns = [f'{loc_prefix}_{column}' for column in columns]
    df.to_csv(metapath / Path(f'{location}.csv'), index=False, header=True, columns=columns)

