import torch
from torch import nn

from .dinov2 import DINOv2


class Model(nn.Module):
    def __init__(self, encoder: str = "vitl"):
        super().__init__()

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

    def forward(self, x: torch.Tensor):
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )

        return features
