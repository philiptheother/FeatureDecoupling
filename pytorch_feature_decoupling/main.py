import pdb
import logging
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imp
import numpy as np

import torch
import torchvision

import config_env as env

parser = argparse.ArgumentParser()
parser.add_argument('--exp',            type=str,   required=True,  default='', help='config file with parameters of the experiment')
parser.add_argument('--evaluate',       type=int,   default=1)
parser.add_argument('--num_workers',    type=int,   default=0,      help='number of data loading workers')
parser.add_argument('--disp_step',      type=int,   default=1,      help='display step during training')
parser.add_argument('--checkpoint',     type=int,   default=0,      help='checkpoint (epoch id) that will be loaded')
args_opt = parser.parse_args()

exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
# if args_opt.semi == -1:
exp_directory = os.path.join("../","_experiments",args_opt.exp)
# else:
#    assert(args_opt.semi>0)
#    exp_directory = os.path.join('.','experiments/unsupervised',args_opt.exp+'_semi'+str(args_opt.semi))

# Load the configuration params of the experiment
logging.info('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
logging.info("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
logging.info("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']

config['disp_step'] = args_opt.disp_step

from dataloader import GenericDataset, DataLoader
dataset_train = GenericDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    random_sized_crop=data_train_opt['random_sized_crop'])
dataset_test = GenericDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    random_sized_crop=data_test_opt['random_sized_crop'])

dloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=data_train_opt['batch_size'],
    unsupervised=data_train_opt['unsupervised'],
    num_workers=args_opt.num_workers,
    shuffle=True)
dloader_test = DataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    unsupervised=data_test_opt['unsupervised'],
    num_workers=args_opt.num_workers,
    shuffle=False)

is_evaluation = True if (args_opt.evaluate==1) else False
if is_evaluation:
    logging.info("### ----- Evaluation: inference only. ----- ###")
else:
    logging.info("### ----- Training: train model. ----- ###")

if torch.cuda.is_available():
    logging.info("### ----- GPU device available, arrays will be copied to cuda. ----- ###")
else:
    logging.info("### ----- GPU device is unavailable, computation will be performed on CPU. ----- ###")

import algorithms as alg
algorithm = getattr(alg, config['algorithm_type'])(config)

if torch.cuda.is_available(): # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint > 0: # load checkpoint
    algorithm.load_checkpoint(args_opt.checkpoint, train= (not is_evaluation))

if not is_evaluation: # train the algorithm
    algorithm.solve(dloader_train, dloader_test)
else:
    algorithm.evaluate(dloader_test) # evaluate the algorithm
