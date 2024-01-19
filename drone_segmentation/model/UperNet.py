import torch
from transformers.models.upernet.modeling_upernet import UperNetHead
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
import timm
from drone_segmentation.model.backbone.cnn_backbone import CNNBackbone

class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, backbone_name='focalnet_base_srf.ms_in1k', upernet_cfg=None):
        super(UperNet, self).__init__()
        self.backbone = CNNBackbone(backbone_name, True)
        self.head = UperNetHead(upernet_cfg, self.backbone.features_dim)

    def forward(self, x):
        features = self.backbone(x)[-4:]
        output = self.head(features)
        return output

                            
if __name__ == '__main__':
    model = UperNet(num_classes=1, backbone='focalnet_base_srf.ms_in1k').eval()
    x = torch.zeros((1,3,512,512))
    print(model(x).shape)
    