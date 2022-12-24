# Author: Emanuele Vivoli
# github-id: emanuelevivoli

# This is the script to create the inversion (using Fresnel) of holographic images

# Even if it could be possible to analyze holographic images 'as it',
# we took into consideration using inversions as they are more suitable for 
# data fusion and synthetic dataset creation

import os
import numpy as np

import matlab.engine
eng = matlab.engine.start_matlab()

from src.utils.const import *


# add paths to files
s = eng.genpath(os.path.join(PWD, 'matlab'))
eng.addpath(s, nargout=0)

###############
# SPECIFICS
###############

locations = ['indoor', 'outdoor']
prefixes = {
    'indoor': ['in'],
    'outdoor': ['']
}
positions = {
    'indoor': ['bas', 'low', 'pad'],
    'outdoor': ['']
}

###############
# INVERSIONS
#   with
# FRESNEL
###############


for location in locations:
    
    names = os.listdir( os.path.join(DATA_RAW, location) )
    
    # create folder if not exists
    if not os.path.exists(os.path.join(DATA_PROC, location)):
        os.makedirs(os.path.join(DATA_PROC, location))  

    # instead of doing this:
    #   plutos = [name for name in names if name.endswith('.log')]
    #   sambas = [name for name in names if name.endswith('.csv')]
    # we just take the names without extention:
    names_list = [name.split('.')[0] for name in names]
    union = set(names_list)
    intersection = set([ el for el in union if names_list.count(el) > 1])
    
    diff = union - intersection

    for name in intersection:
        pluto = f'{name}.log'
        samba = f'{name}.csv'

        for prefix in prefixes[location]:
            try:
                eng.workspace['pluto']= os.path.join(DATA_RAW, location, pluto)
                eng.workspace['trace']= os.path.join(DATA_RAW, location, samba)

                eng.eval(f"[F,FI,FQ,P_X,P_Y,P_MOD,P_PHASE] = merge_acquisition(trace, pluto);",nargout=0)
                # F = eng.workspace['F']
                # P_X = eng.workspace['P_X']
                # P_Y = eng.workspace['P_Y']
                # P_I = eng.workspace['P_MOD']
                # P_Q = eng.workspace['P_PHASE']

                eng.eval(f"[MO, PH, H, Hfill] = fast_generate_hologram(F, FI, FQ, 5, P_X, P_Y, P_MOD, P_PHASE, 2, 3);",nargout=0)

                # MO = eng.workspace['MO']
                # PH = eng.workspace['PH']
                # H = eng.workspace['H']
                # Hfill = eng.workspace['Hfill']

                #!################
                #! VARIABLES
                #!################

                eng.workspace['epsilon'] = 4 if location == 'outdoor' else 1
                eng.eval("epsilon = double(epsilon);", nargout=0)
                eng.workspace['in_line'] = 0
                eng.eval("in_line = double(in_line);", nargout=0)
                
                #!DAFAULT
                # eng.workspace['min'] = 0
                # eng.eval("min = double(min)/1e2;", nargout=0)
                # eng.workspace['max'] = 40
                # eng.eval("max = double(max)/1e2;", nargout=0)
                # eng.workspace['step'] = 100
                # eng.eval("step = double(step);", nargout=0)
                # eng.workspace['pad_size'] = 1000
                # eng.eval("pad_size = double(pad_size);", nargout=0)
                # eng.workspace['scale_factor'] = 1
                # eng.eval("scale_factor = double(scale_factor);", nargout=0)
                # eng.workspace['resizeFlag'] = 0
                # eng.eval("resizeFlag = double(resizeFlag);", nargout=0)
                
                # inversion algorithm can be one of:
                # - Fresnel
                # - Conv
                # - AngSpec
                
                eng.eval("[Inv] = fast_inversion(Hfill, F, epsilon, 'Fresnel', in_line);", nargout=0)
                eng.eval("Inv_app = cell2mat(Inv);", nargout=0)
                Inv_app = eng.workspace['Inv_app']
                Inv = np.asarray(Inv_app, dtype = 'complex_')
                np.save(file=os.path.join(DATA_PROC, location, f'{name}_0.npy'), arr=Inv)

                eng.eval("[Inv180] = fast_inversion(Hfill.*exp(1i*pi), F, epsilon, 'Fresnel', in_line);", nargout=0)
                eng.eval("Inv180_app = cell2mat(Inv180);", nargout=0)
                Inv180_app = eng.workspace['Inv180_app']
                Inv180 = np.asarray(Inv180_app, dtype = 'complex_')
                np.save(file=os.path.join(DATA_PROC, location, f'{name}_180.npy'), arr=Inv180)

            except Exception as e:
                print(e)

