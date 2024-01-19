import timm
import torch.nn as nn
import math
from einops import rearrange, reduce, repeat
import torch

def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False
        
class CNNBackbone(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.dino_backbone = timm.create_model('timm/vit_small_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_pad=True)
        self.features_dim = [x['num_chs'] for x in self.backbone.feature_info[-4:]]
        self.projector = nn.Conv2d(self.dino_backbone.num_features+self.features_dim[-1], self.features_dim[-1], 1, 1, 0)
        freeze_model_params(self.dino_backbone)
    def forward(self, x):
        features = self.backbone(x)
        N, C, H, W = features[-1].shape
        dino_feature = self.dino_backbone.forward_features(x)
        dino_feature = dino_feature[:,1:,:]
        R = int(math.sqrt(dino_feature.shape[1]))
        dino_feature = rearrange(dino_feature, 'n (l1 l2) d -> n d l1 l2', l1=R, l2=R)
        dino_feature = nn.functional.interpolate(dino_feature, (H, W))
        features[-1] = torch.concat([dino_feature, features[-1]], dim=1)
        features[-1] = self.projector(features[-1])
        return features
    