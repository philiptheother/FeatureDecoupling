import pdb

import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

class LinearTransformationNorm(nn.Module):
    def __init__(self, opt):
        super(LinearTransformationNorm, self).__init__()

        self.norm = nn.Sequential(
            nn.Linear(opt['num_feat'], opt['low_dim']),
            Normalize(2),
        )

    def forward(self, feat):
        return self.norm(feat)

def create_model(opt):
    return LinearTransformationNorm(opt)
