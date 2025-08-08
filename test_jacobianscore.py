import json

from load_cifer10 import load_cifar10
from models.simclr import SimCLR
from torch import nn
import numpy as np


with open('utils/mlp_bench.json') as f:
    mlp_bench = json.load(f)

rando_arch = mlp_bench[3]
print(rando_arch)


# create Simclr architecture


model = SimCLR(rando_arch)
l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]


print(l)



