import os
import numpy as np

def twoclass(label:str)->int:
    # 1 -> mine
    # 0 -> other
    return 1 if label == 'mine' else 0

def threeclass(label:str)->int:
    # 2 -> archeology 
    # 1 -> mine
    # 0 -> other
    return 1 if label == 'mine' else 0 if label == 'archeology' else 2

def fgclass(label:str)->int:
    # 14 -> knife
    # 13 -> terracotta
    # ...
    # 1 -> pmn1
    # 0 -> none
    return int(label)-1
    
def task_manager(task_name, row):
        
    if task_name == 'binary':
        label = twoclass(row['in_name'])

    elif task_name == 'trinary':
        label = threeclass(row['in_name'])

    elif task_name == 'multi':
        label = fgclass(row['in_id'])
        
    else:
        label = None
    
    return label

def get_holo_noise(indoorpath, in_file_name, in_id, size):
    try: holo_noise = np.load(indoorpath / f"{in_file_name}.npy".replace(str(in_id).zfill(2), '00'))
    except:
        try: holo_noise = np.load(indoorpath / f"{in_file_name.replace('bas', 'low')}.npy".replace(str(in_id).zfill(2), '00'))
        except:
            print(f"no noise signal {in_file_name.replace(str(in_id).zfill(2), '00')} exists for {in_file_name}")
            holo_noise = np.zeros(size)

    return holo_noise

def read_scans(indoor_dir, outdoor_dir):
    import os

    indoor_scans = []
    outdoor_scans = []

    for file_name in os.listdir(indoor_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(indoor_dir, file_name)
            scan = np.load(file_path)
            indoor_scans.append(scan)

    for file_name in os.listdir(outdoor_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(outdoor_dir, file_name)
            scan = np.load(file_path)
            outdoor_scans.append(scan)
    
    return indoor_scans, outdoor_scans

def if_null_create(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def check_task_classes(cfg):
    if cfg.data.task == 'binary':
        if cfg.model.num_classes != 2:
            raise ValueError("The model is not configured for binary classification. Check the configuration file.")
    elif cfg.data.task == 'trinary':
        if cfg.model.num_classes != 3:
            raise ValueError("The model is not configured for ternary classification. Check the configuration file.")
    elif cfg.data.task == 'multi':
        print(cfg.model.num_classes)
        if cfg.model.num_classes != 13:
            raise ValueError("The model is not configured for multi classification. Check the configuration file.")
    else:
        raise ValueError("The task is not recognized. Check the configuration file.")
    
    return None