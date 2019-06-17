import pdb

import torch
import torch.nn as nn

class AlexNetClassifier(nn.Module):
    def __init__(self, opt):
        super(AlexNetClassifier, self).__init__()

        num_classes = opt['num_classes']
        num_fc7_feats = opt['num_feat'] if 'num_feat' in opt else 2048
        self.fc_classifier = nn.Sequential(nn.Linear(num_fc7_feats, num_classes),)

    def forward(self, feat):
        return self.fc_classifier(feat)

def create_model(opt):
    return AlexNetClassifier(opt)
