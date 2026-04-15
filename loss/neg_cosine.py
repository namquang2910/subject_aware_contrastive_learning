import torch
import torch.nn as nn
import torch.nn.functional as F

class NegCosine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, p, z):
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()