'''
This file is from
https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/alias_multinomial.py
and
https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCEAverage.py
'''
import pdb

import math
import numpy as np

import torch
from torch.autograd import Function
from torch import nn

class AliasMethod(object):
    '''
        From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        print("Moving AliasMethod To CUDA.")
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        '''
            Draw N samples from multinomial
        '''
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()

        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0)
        inputSize = memory.size(1)

        # sample positives & negatives
        idx.select(1,0).copy_(y.data)

        # sample correspoinding weights
        weight = torch.index_select(memory, 0, idx.view(-1))
        weight.resize_(batchSize, K+1, inputSize)

        # inner product
        out = torch.bmm(weight, x.data.resize_(batchSize, inputSize, 1))
        out.div_(T).exp_() # batchSize * self.K+1
        x.data.resize_(batchSize, inputSize)

        if Z < 0:
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        out.div_(Z).resize_(batchSize, K+1)

        self.save_for_backward(x, memory, y, weight, out, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        
        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)

        gradOutput.data.resize_(batchSize, 1, K+1)
        
        # gradient of linear
        gradInput = torch.bmm(gradOutput.data, weight)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None, None

class NCEAverage(nn.Module):
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.K = K

        self.register_buffer('params',torch.tensor([K, T, -1, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
 
    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (self.K+1)).view(batchSize, -1)
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        return out

    def cuda(self, device=None):
        print("Moving NCE To CUDA.")
        self.multinomial.cuda()
        return super(NCEAverage, self).cuda(device)

def create_model(opt):
    low_dim = opt['low_dim']
    ndata = opt['ndata']
    nce_k = opt['nce_k']
    nce_t = opt['nce_t']
    nce_m = opt['nce_m']
    assert(nce_k>0)
    return NCEAverage(low_dim, ndata, nce_k, nce_t, nce_m)
