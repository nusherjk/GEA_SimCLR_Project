import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class EncoderResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        base_encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])  # Remove FC
        # self.projector = MLPProjector(projector_config)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        return h
