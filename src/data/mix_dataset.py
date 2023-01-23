import os
import numpy as np

# variables




locations = ['indoor', 'outdoor']

for location in locations:
    scans = []
    scan_dir = 'path_to_scans'
    for file_name in os.listdir(scan_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(scan_dir, file_name)
            scan = np.load(file_path)
            scans.append(scan)

    scans = np.stack(scans)
    np.save('path_to_save_scans', scans)