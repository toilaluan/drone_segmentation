from transformers import UperNetConfig, UperNetForSemanticSegmentation
import torch.nn as nn
import torch

class UperNet_HF(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.model = UperNetForSemanticSegmentation.from_pretrained(pretrained_name, num_labels = 1, ignore_mismatched_sizes=True)
    
    def forward(self, x):
        out = self.model(x)
        return out.logits
    
    
if __name__ == '__main__':
    x = torch.zeros((1,3,512,512))
    model = UperNet_HF('openmmlab/upernet-convnext-base')
    out = model(x)
    print(out.shape)