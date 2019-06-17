"""Define a generic class for training and testing learning algorithms."""
import pdb
import logging

import os
import os.path
import datetime
import time
import imp
import math
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim

import config_env as env
import utils

class Algorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir'])
        self.set_log_file_handler()

        self.logger.info('Algorithm options %s' % opt)
        self.opt = opt
        self.init_all_networks()
        self.init_all_criterions()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}

        self.keep_best_model_metric_name = opt['best_metric'] if ('best_metric' in opt) else None

    def set_experiment_dir(self,directory_path):
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)

    def set_log_file_handler(self):
        logging.getLogger().handlers = []
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")

        strHandler = logging.StreamHandler()
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)
        now_str = datetime.datetime.now().strftime('%m%d_%H%M%S')
        log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.log')
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)

    def init_all_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}

        for name_net, def_net in networks_defs.items(): # def_net: {'def_file','pretrained','opt','optim_params'{}
            self.logger.info('Set network %s' % name_net)
            def_file = def_net['def_file']
            net_opt = def_net['opt']
            self.optim_params[name_net] = def_net['optim_params'] if ('optim_params' in def_net) else None
            pretrained_path = def_net['pretrained'] if ('pretrained' in def_net) else None
            self.networks[name_net] = self.init_network(def_file, net_opt, pretrained_path, name_net)

    def init_network(self, net_def_file, net_opt, pretrained_path, name_net):
        self.logger.info('==> Initiliaze network %s from file %s with opts: %s' % (name_net, net_def_file, net_opt))
        if (not os.path.isfile(net_def_file)):
            raise ValueError('Non existing file: {0}'.format(net_def_file))

        network = imp.load_source("",net_def_file).create_model(net_opt)
        if pretrained_path != None:
            self.load_pretrained(network, pretrained_path)

        return network

    def load_pretrained(self, network, pretrained_path):
        self.logger.info('==> Load pretrained parameters from file %s:' % (pretrained_path))

        assert(os.path.isfile(pretrained_path))
        if "cpu"==env.PROCESS_UNIT:
            pretrained_model = torch.load(pretrained_path, map_location='cpu')
        elif "gpu"==env.PROCESS_UNIT:
            pretrained_model = torch.load(pretrained_path)
        else:
            raise ValueError('Process unit platform is not specified: {0}'.format(env.PROCESS_UNIT))
        
        if 'module' in list(pretrained_model['network'].keys())[0]:
            self.logger.info('==> Network keys in pre-trained file %s contain \"module\"' % (pretrained_path))
            from collections import OrderedDict
            pretrained_model_nomodule = OrderedDict()
            for key, value in pretrained_model['network'].items():
                key_nomodule = key[7:] # remove module
                pretrained_model_nomodule[key_nomodule] = value
        else:
            pretrained_model_nomodule = pretrained_model['network']

        if pretrained_model_nomodule.keys() == network.state_dict().keys():
            network.load_state_dict(pretrained_model_nomodule)
        else:
            self.logger.info('==> WARNING: network parameters in pre-trained file %s do not strictly match' % (pretrained_path))
            network.load_state_dict(pretrained_model_nomodule, strict=False)

    def init_all_optimizers(self):
        self.optimizers = {}

        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams != None:
                self.optimizers[key] = self.init_optimizer(
                        self.networks[key], oparams, key)

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.logger.info('Initialize optimizer: %s with params: %s for network: %s' % (optim_type, optim_opts, key))
        if optim_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                        betas=optim_opts['beta'])
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                momentum=optim_opts['momentum'],
                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
                weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)

        return optimizer

    def init_all_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for name_cri, def_cri in criterions_defs.items():
            crit_type = def_cri['ctype']
            crit_opt = def_cri['opt'] if ('opt' in def_cri) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (name_cri, crit_type, crit_opt))
            self.criterions[name_cri] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        if 'CrossEntropyLoss'==ctype:
            if copt is not None:
                return getattr(nn, ctype)(**copt)
            else:
                return getattr(nn, ctype)()
        elif 'MSELoss'==ctype:
            return getattr(nn, ctype)()
        elif 'NCEAverage'==ctype:
            from architectures.NCEAverage import create_model as createModelNCEAverage
            return createModelNCEAverage(copt['net_opt'])
        elif 'NCECriterion'==ctype:
            return self.init_network(copt['def_file'], copt['net_opt'], None, ctype)
        else:
            return getattr(nn, ctype)(copt)


    def load_to_gpu(self):
        for key, net in self.networks.items():
            self.networks[key] = torch.nn.DataParallel(net).cuda()

        for key, criterion in self.criterions.items():
            self.criterions[key] = criterion.cuda()

        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.cuda()

    def save_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue
            self.save_network(key, epoch, suffix=suffix)
            self.save_optimizer(key, epoch, suffix=suffix)

        for key, net in self.criterions.items():
            if 'nce_average'==key:
                self.save_criterion(key, epoch, suffix=suffix)

    def load_checkpoint(self, epoch, train=True, suffix=''):
        self.logger.info('Load checkpoint of epoch %d' % (epoch))

        for key, net in self.networks.items(): # Load networks
            if self.optim_params[key] == None: continue
            self.load_network(key, epoch,suffix)

        for key, net in self.criterions.items():
            if 'nce_average'==key:
                self.load_criterion(key, epoch,suffix)

        if train: # initialize and load optimizers
            self.init_all_optimizers()
            for key, net in self.networks.items():
                if self.optim_params[key] == None: continue
                self.load_optimizer(key, epoch,suffix)

        self.curr_epoch = epoch

    def delete_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue

            filename_net = self._get_net_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_net): os.remove(filename_net)

            filename_optim = self._get_optim_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_optim): os.remove(filename_optim)

        for key, net in self.criterions.items():
            if 'nce_average'==key:
                filename_cri = self._get_cri_checkpoint_filename(key, epoch)+suffix
                if os.path.isfile(filename_cri): os.remove(filename_cri)

    def save_network(self, net_key, epoch, suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
        state = {'epoch': epoch,'network': self.networks[net_key].state_dict()}
        torch.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
        state = {'epoch': epoch,'optimizer': self.optimizers[net_key].state_dict()}
        torch.save(state, filename)

    def save_criterion(self, net_key, epoch, suffix=''):
        assert(net_key in self.criterions)
        filename = self._get_cri_checkpoint_filename(net_key, epoch)+suffix
        state = {'epoch': epoch,'criterion': self.criterions[net_key],}
        torch.save(state, filename)

    def load_network(self, net_key, epoch,suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            if "cpu"==env.PROCESS_UNIT:
                checkpoint = torch.load(filename, map_location='cpu')
            elif "gpu"==env.PROCESS_UNIT:
                checkpoint = torch.load(filename)
            else:
                raise ValueError('Process unit platform is not specified: {0}'.format(env.PROCESS_UNIT))

            if 'module' in list(checkpoint['network'].keys())[0] and 'module' not in list(self.networks[net_key].state_dict().keys())[0]:
                self.logger.info('Network keys in checkpoint file %s contain module' % (filename))
                from collections import OrderedDict
                pretrained_model = OrderedDict()
                for key, value in checkpoint['network'].items():
                    key_nomodule = key[7:] # remove module
                    pretrained_model[key_nomodule] = value

            elif 'module' not in list(checkpoint['network'].keys())[0] and 'module' in list(self.networks[net_key].state_dict().keys())[0]:
                self.logger.info('Network keys contain module')
                from collections import OrderedDict
                pretrained_model = OrderedDict()
                for key, value in checkpoint['network'].items():
                    key_module = "module."+key # add module
                    pretrained_model[key_module] = value

            else:
                pretrained_model = checkpoint['network']

            if self.networks[net_key].state_dict().keys() == pretrained_model.keys():
                self.networks[net_key].load_state_dict(pretrained_model)
            else:
                self.logger.info('==> WARNING: network parameters in checkpoint file %s do not strictly match' % (filename))
                self.networks[net_key].load_state_dict(pretrained_model, strict=False)

    def load_optimizer(self, net_key, epoch,suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            if "cpu"==env.PROCESS_UNIT:
                checkpoint = torch.load(filename, map_location='cpu')
            elif "gpu"==env.PROCESS_UNIT:
                checkpoint = torch.load(filename)
            else:
                raise ValueError('Process unit platform is not specified: {0}'.format(env.PROCESS_UNIT))
            self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])

    def load_criterion(self, net_key, epoch,suffix=''):
        assert(net_key in self.criterions)
        filename = self._get_cri_checkpoint_filename(net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            if "cpu"==env.PROCESS_UNIT:
                checkpoint = torch.load(filename, map_location='cpu')
            elif "gpu"==env.PROCESS_UNIT:
                checkpoint = torch.load(filename)
            else:
                raise ValueError('Process unit platform is not specified: {0}'.format(env.PROCESS_UNIT))
            self.criterions[net_key] = checkpoint['criterion']

    def _get_net_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_net_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_optim_epoch'+str(epoch))

    def _get_cri_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_cri_epoch'+str(epoch))

    def solve(self, data_loader_train, data_loader_test):
        self.max_num_epochs = self.opt['max_num_epochs']
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()

        eval_stats  = {}
        train_stats = {}
        self.init_record_of_best_model()
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            self.logger.info("\n")
            self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch+1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            self.logger.info('==> Training stats: %s' % (train_stats))

            self.save_checkpoint(self.curr_epoch+1) # create a checkpoint in the current epoch
            if start_epoch != self.curr_epoch: # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)

            if (data_loader_test is not None):
                self.logger.info("\n")
                eval_stats = self.evaluate(data_loader_test)
                self.logger.info('==> Evaluation stats: %s' % (eval_stats))
                self.keep_record_of_best_model(eval_stats, self.curr_epoch)

        self.print_eval_stats_of_best_model()
        self.logger.info("\n")

    def run_train_epoch(self, data_loader, epoch):
        self.logger.info('Training: %s' % os.path.basename(self.exp_dir))
        self.logger.info('==> Dataset: %s [%s images]'%(data_loader.dataset.name, str(len(data_loader.dataset))))
        self.logger.info("==> Iteration steps in one epoch: %d [batch size %d]"%(len(data_loader), data_loader.batch_size))

        for key, network in self.networks.items():
            if self.optimizers[key] == None: network.eval()
            else: network.train()

        disp_step   = self.opt['disp_step'] if ('disp_step' in self.opt) else 1
        train_stats = utils.DAverageMeter()

        for idx, batch in enumerate(tqdm(data_loader(epoch), total=len(data_loader))):

            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx+1) % disp_step == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' % (epoch+1, idx+1, len(data_loader), train_stats.average()))

        return train_stats.average()

    def evaluate(self, dloader):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))
        self.logger.info('==> Dataset: %s [%d images]' % (dloader.dataset.name, len(dloader.dataset)))
        self.logger.info("==> Iteration steps in one epoch: %d [batch size %d]"%(len(dloader), dloader.batch_size))
    
        for key, network in self.networks.items():
            network.eval()

        eval_stats = utils.DAverageMeter()

        for idx, batch in enumerate(tqdm(dloader(), total=len(dloader))):
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)

        self.logger.info('==> Results: %s' % eval_stats.average())

        return eval_stats.average()

    def adjust_learning_rates(self, epoch):
        # filter out the networks that are not trainable and that do
        # not have a learning rate Look Up Table (LUT_lr) in their optim_params
        optim_params_filtered = {k:v for k,v in self.optim_params.items()
            if (v != None and ('LUT_lr' in v))}

        for key, oparams in optim_params_filtered.items():
            LUT = oparams['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
            self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in self.optimizers[key].param_groups:
                param_group['lr'] = lr

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:
            metric_name = self.keep_best_model_metric_name
            if (metric_name not in eval_stats):
                raise ValueError('The provided metric {0} for keeping the best model is not computed by the evaluation routine.'.format(metric_name))
            metric_val = eval_stats[metric_name]
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                self.save_checkpoint(self.curr_epoch+1, suffix='.best')
                if self.best_epoch is not None:
                    self.delete_checkpoint(self.best_epoch+1, suffix='.best')
                self.best_epoch = current_epoch
                self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info('==> Best results w.r.t. %s metric: epoch: %d - %s' % (metric_name, self.best_epoch+1, self.best_stats))


    # FROM HERE ON ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE CLASS
    # THAT INHERITS THE Algorithms CLASS
    def train_step(self, batch):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
            * Return a dictionary with the computed losses and any other desired
                stats. The key names on the dictionary can be arbitrary.
        """
        pass

    def evaluation_step(self, batch):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
            * Return a dictionary with the computed losses the evaluation
                metrics for that batch. The key names on the dictionary can be
                arbitrary.
        """
        pass

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensors = {}
