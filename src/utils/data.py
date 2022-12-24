import numpy as np

def twoclass(label:str)->int:
    # 1 -> mine
    # 0 -> other
    return 1 if label == 'mine' else 0

def threeclass(label:str)->int:
    # 2 -> archeology 
    # 1 -> mine
    # 0 -> other
    return 1 if label == 'mine' else 2 if label == 'archeology' else 0

def fgclass(label:str)->int:
    # 14 -> knife
    # 13 -> terracotta
    # ...
    # 1 -> pmn1
    # 0 -> none
    return int(label)
    
def task_manager(task_name, row):
        
    if task_name == '2-class':
        label = twoclass(row['in_name'])

    elif task_name == '3-class':
        label = threeclass(row['in_name'])

    elif task_name == 'fg-class':
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