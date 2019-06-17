import pdb
import time

import numpy as np

import torch
import torch.nn as nn

from . import Algorithm
import utils

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class FeatureClassificationModel(Algorithm):
    def __init__(self, opt):
        self.out_feat_keys = opt['out_feat_keys']
        Algorithm.__init__(self, opt)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start
        #********************************************************

        start = time.time()
        #************ FORWARD PROPAGATION ***********************
        finetune_feat_extractor = (self.optimizers['feat_extractor'] is not None) if do_train else False
        if do_train:
            if finetune_feat_extractor:
                for param in self.networks['feat_extractor'].parameters():
                    param.requires_grad = True
            else:
                for param in self.networks['feat_extractor'].parameters():
                    param.requires_grad = False
                self.networks['feat_extractor'].eval()
            for param in self.networks['classifier'].parameters():
                param.requires_grad = True
        else:
            for param in self.networks['feat_extractor'].parameters():
                param.requires_grad = False
            self.networks['feat_extractor'].eval()
            for param in self.networks['classifier'].parameters():
                param.requires_grad = False
            self.networks['classifier'].eval()

        with torch.set_grad_enabled(finetune_feat_extractor):
            feat_var = self.networks['feat_extractor'](dataX, out_feat_keys=self.out_feat_keys)

        with torch.set_grad_enabled(do_train):
            pred_var = self.networks['classifier'](feat_var)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        record = {}
        if isinstance(pred_var, (list, tuple)):
            loss_total = None
            for i in range(len(pred_var)):
                loss_this = self.criterions['loss'](pred_var[i], labels)
                loss_total = loss_this if (loss_total is None) else (loss_total + loss_this)
                record['prec1_conv'+str(1+i)] = accuracy(pred_var[i], labels, topk=(1,))[0].item()
        else:
            loss_total = self.criterions['loss'](pred_var, labels)
            record['prec1'] = accuracy(pred_var, labels, topk=(1,))[0].item()
        record['loss'] = loss_total.item()
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            self.optimizers['classifier'].zero_grad()
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].zero_grad()

            loss_total.backward()
            self.optimizers['classifier'].step()
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record
