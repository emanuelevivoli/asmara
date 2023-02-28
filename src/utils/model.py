import torch.nn as nn

import importlib
from omegaconf import DictConfig, OmegaConf
from src.utils.const import BASEPATH

import sys

sys.path.append(BASEPATH)

def instantiate_from_config(cfg: DictConfig) -> nn.Module:
    # Extract the module name and class name from the config
    module_name, class_name = cfg._target_.rsplit(".", 1)
    
    # Import the module dynamically using importlib
    module = importlib.import_module(module_name)
    
    # Get a reference to the class object using getattr
    class_ = getattr(module, class_name)

    # get the constructor arguments from the configuration object
    constructor_args = OmegaConf.to_container(cfg, resolve=True)

    # remove the '_target_' key from the constructor arguments
    constructor_args.pop("_target_")

    # instantiate the class with the constructor arguments
    instance = class_(**constructor_args)

    return instance
