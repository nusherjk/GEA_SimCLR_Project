import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])  # Remove FC
        # self.projector = MLPProjector(projector_config)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        return h
