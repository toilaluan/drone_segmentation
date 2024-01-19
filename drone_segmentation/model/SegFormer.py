from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import torch

class SegFormer(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(pretrained_name, num_labels = 4, ignore_mismatched_sizes=True)
    
    def forward(self, x):
        out = self.model(x)
        return out.logits
    
    
if __name__ == '__main__':
    x = torch.zeros((1,3,512,512))
    model = SegFormer('nvidia/segformer-b0-finetuned-ade-512-512')
    out = model(x)
    print(out.shape)