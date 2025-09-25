import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_encoder = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])  # Remove FC
        # self.projector = MLPProjector(projector_config)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        return h
