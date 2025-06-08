
import torch
import torch.nn as nn
import torchvision.models as models
from .mlp_projector import MLPProjector

class SimCLR(nn.Module):
    def __init__(self, projector_config):
        super().__init__()
        base_encoder = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])  # Remove FC
        self.projector = MLPProjector(projector_config)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return z
