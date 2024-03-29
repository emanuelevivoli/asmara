from omegaconf import OmegaConf

from src.models.nets import *
from src.utils.const import BASEPATH
from src.utils.spec import CLASS_NUMBER
from src.utils.model import instantiate_from_config

from torchsummary import summary

# get as input the name of the model and the task
# and return the model with the correct number of classes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", '-m', type=str, default="simplevit", choices=["simplevit", "unet", "simplevit3d", "unet3d", "resnet", "effnet"])
parser.add_argument("--task", '-t', type=str, default="binary", choices=["binary", "trinary", "multi"])
parser.add_argument("--dataset", '-d', type=str, default="binary", choices=["holograms", "inversions"])
args = parser.parse_args()

cfg = OmegaConf.load(f'{BASEPATH}/src/config/default.yaml')

cfg.model.name = args.model
cfg.data.task = args.task
cfg.data.dataset = args.dataset

model_cfg = OmegaConf.load(f"{BASEPATH}/src/config/models/{cfg.model.name}.yaml")
cfg.model = OmegaConf.merge(cfg.model, model_cfg)

cfg.model.num_classes = CLASS_NUMBER[cfg.data.task]

# Initialize the network
model = instantiate_from_config(cfg.model, cfg.optimizer)
model.cuda()

# Print the size of the model's parameters
print(f"Model parameters size: {sum(p.numel() for p in model.parameters())}")
# Print the amount of available GPU memory
device = torch.device("cuda")
print(f"Available GPU memory: {torch.cuda.get_device_properties(device).total_memory / (1024**2)} MB")
data_size = (1, 60, 60) if cfg.data.dataset == "holograms" else (1, 41, 60, 60)
summary(model, data_size)
