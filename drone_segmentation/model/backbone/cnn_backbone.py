import timm
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.features_dim = [x['num_chs'] for x in self.backbone.feature_info[-4:]]
    def forward(self, x):
        return self.backbone(x)
    