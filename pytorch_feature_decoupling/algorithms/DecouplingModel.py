import pdb

import os
import time
import numpy as np

import torch

from . import Algorithm

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

class DecouplingModel(Algorithm):
    def __init__(self, opt):
        self.lambda_loss = opt['lambda_loss']
        self.gama = opt['gama']
        Algorithm.__init__(self, opt)

    def loadPUImageProb(self):
        with open(os.path.join(self.exp_dir, 'prob', 'prob.dat'), 'r') as file_input:
            train_prob_str = file_input.readlines()
            train_prob = [float(i_prob_str.rstrip('\n')) for i_prob_str in train_prob_str]
        self.train_weight = [1.0 if 0==i%4 else 1-train_prob[i]**self.gama for i in range(len(train_prob))]

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['index'] = torch.LongTensor()
        self.tensors['index_index'] = torch.LongTensor()
        self.tensors['labels'] = torch.LongTensor()
        self.loadPUImageProb()
        
    def train_step(self, batch):
        start = time.time()
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        dataX_90 = torch.flip(torch.transpose(self.tensors['dataX'],2,3),[2])
        dataX_180 = torch.flip(torch.flip(self.tensors['dataX'],[2]),[3])
        dataX_270 = torch.transpose(torch.flip(self.tensors['dataX'],[2]),2,3)

        dataX = torch.stack([self.tensors['dataX'], dataX_90, dataX_180, dataX_270], dim=1)
        batch_size, rotations, channels, height, width = dataX.size()
        dataX = dataX.view([batch_size*rotations, channels, height, width])

        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        self.tensors['index'].resize_(batch[2].size()).copy_(batch[2])
        labels = self.tensors['labels']
        index = self.tensors['index']
        
        idx_train = 4*batch[2].numpy()
        idx_train[1::4] += 1
        idx_train[2::4] += 2
        idx_train[3::4] += 3
        #********************************************************
        batch_load_time = time.time() - start

        start = time.time()
        #************ FORWARD THROUGH NET ***********************
        for _, network in self.networks.items():
            for param in network.parameters():
                param.requires_grad = True

        with torch.set_grad_enabled(True):
            feature = self.networks['feature'](dataX)
            feature_rot, feature_invariance = torch.split(feature, 2048, dim=1)

            pred = self.networks['classifier'](feature_rot)

            feature_invariance_instance = feature_invariance[0::4,:] + feature_invariance[1::4,:] + feature_invariance[2::4,:] + feature_invariance[3::4,:]
            feature_invariance_instance = torch.mul(feature_invariance_instance, 0.25)
            feature_nce_norm = self.networks['norm'](feature_invariance_instance)
        
        with torch.set_grad_enabled(False):
            self.tensors['index_index'].resize_(torch.Size([int(index.size(0)/4)])).copy_(index[0::4])
            index_instance = self.tensors['index_index']
            feature_invariance_instance_mean = torch.unsqueeze(feature_invariance_instance,1).expand(-1,4,-1).clone()
            feature_invariance_instance_mean = feature_invariance_instance_mean.view(4*len(feature_invariance_instance),2048)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        weight = torch.tensor(np.array(self.train_weight)[idx_train], dtype=torch.float, device=labels.device, requires_grad=False)
        with torch.set_grad_enabled(True):
            loss_cls_each = self.criterions['loss_cls'](pred, labels)
            if self.curr_epoch<210:
                loss_cls = torch.sum(loss_cls_each)/loss_cls_each.shape[0]
            else:
                loss_cls = torch.dot(loss_cls_each,weight)/loss_cls_each.shape[0]

            loss_mse = self.criterions['loss_mse'](feature_invariance, feature_invariance_instance_mean)

            output_nce = self.criterions['nce_average'](feature_nce_norm, index_instance)
            loss_nce = self.criterions['nce_criterion'](output_nce, index_instance)

            loss_total = self.lambda_loss['cls']*loss_cls + self.lambda_loss['mse']*loss_mse + self.lambda_loss['nce']*loss_nce

        record = {}
        record['prec_cls'] = accuracy(pred, labels, topk=(1,))[0].item()

        record['loss'] = loss_total.item()
        record['loss_cls'] = loss_cls.item()
        record['loss_mse'] = loss_mse.item()
        record['loss_nce'] = loss_nce.item()
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        self.optimizers['feature'].zero_grad()
        self.optimizers['classifier'].zero_grad()
        self.optimizers['norm'].zero_grad()
        loss_total.backward()
        self.optimizers['feature'].step()
        self.optimizers['classifier'].step()
        self.optimizers['norm'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record

    def evaluation_step(self, batch):
        start = time.time()
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        dataX_90 = torch.flip(torch.transpose(self.tensors['dataX'],2,3),[2])
        dataX_180 = torch.flip(torch.flip(self.tensors['dataX'],[2]),[3])
        dataX_270 = torch.transpose(torch.flip(self.tensors['dataX'],[2]),2,3)

        dataX = torch.stack([self.tensors['dataX'], dataX_90, dataX_180, dataX_270], dim=1)
        batch_size, rotations, channels, height, width = dataX.size()
        dataX = dataX.view([batch_size*rotations, channels, height, width])

        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        #********************************************************
        batch_load_time = time.time() - start

        start = time.time()
        #************ FORWARD THROUGH NET ***********************
        for _, network in self.networks.items():
            for param in network.parameters():
                param.requires_grad = False

        with torch.set_grad_enabled(False):
            feature = self.networks['feature'](dataX)
            feature_rot, feature_invariance = torch.split(feature, 2048, dim=1)
            pred_rot = self.networks['classifier'](feature_rot)
            pred_inv = self.networks['classifier'](feature_invariance)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        with torch.set_grad_enabled(False):
            loss_rot_each = self.criterions['loss_cls'](pred_rot, self.tensors['labels'])
            loss_inv_each = self.criterions['loss_cls'](pred_inv, self.tensors['labels'])
            loss_rot = torch.sum(loss_rot_each)/loss_rot_each.shape[0]
            loss_inv = torch.sum(loss_inv_each)/loss_inv_each.shape[0]   
        record = {}
        record['prec_rot'] = accuracy(pred_rot, self.tensors['labels'], topk=(1,))[0].item()
        record['prec_inv'] = accuracy(pred_inv, self.tensors['labels'], topk=(1,))[0].item()
        record['loss_rot'] = loss_rot.item()
        record['loss_inv'] = loss_inv.item()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record
