
import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        for layer_cfg in config:
            layers.append(nn.Linear(layer_cfg['in'], layer_cfg['out']))
            if 'activation' in layer_cfg:
                if layer_cfg['activation'] == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif layer_cfg['activation'] == 'silu':
                    layers.append(nn.SiLU(inplace=True))
                elif layer_cfg['activation'] == 'tanh':
                    layers.append(nn.Tanh())
            if layer_cfg.get('batchnorm', False):
                layers.append(nn.BatchNorm1d(layer_cfg['out']))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
