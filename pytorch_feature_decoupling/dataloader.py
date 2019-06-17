import pdb

import os
import sys
import io
import random

from tqdm import tqdm
import numpy as np
from PIL import Image
import csv

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt
from torch.utils.data.dataloader import default_collate

try:
    import config_env as env
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config_env as env

class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'data', 'vision', 'torralba', 'deeplearning', 'images256')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')

        self.transform = transform
        self.target_transform = target_transform
        with open(split_csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            self.img_files = []
            self.labels = []
            for row in reader:
                self.img_files.append(row[0])
                self.labels.append(int(row[1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_folder, self.img_files[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop

        if self.dataset_name=='imagenet':
            assert(self.split=='train' or self.split=='val')

            if self.split!='train':
                transforms_list_augmentation = [transforms.Resize(256),
                                                transforms.CenterCrop(224)]
            else:
                if self.random_sized_crop:
                    transforms_list_augmentation = [transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip()]
                else:
                    transforms_list_augmentation = [transforms.Resize(256),
                                                    transforms.RandomCrop(224),
                                                    transforms.RandomHorizontalFlip()]

            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            transforms_list_normalize = [transforms.ToTensor(),
                                         transforms.Normalize(mean=self.mean_pix, std=self.std_pix)]

            self.transform_augmentation_normalize = transforms.Compose(transforms_list_augmentation+transforms_list_normalize)
            split_data_dir = env.IMAGENET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform_augmentation_normalize)

        elif self.dataset_name=='places205':
            if self.split!='train':
                transforms_list_augmentation = [transforms.CenterCrop(224)]
            else:
                if self.random_sized_crop:
                    transforms_list_augmentation = [transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip()]
                else:
                    transforms_list_augmentation = [transforms.RandomCrop(224),
                                                    transforms.RandomHorizontalFlip()]

            # ImageNet mean and var for ImageNet pretrained models.
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            transforms_list_normalize = [transforms.ToTensor(),
                                         transforms.Normalize(mean=self.mean_pix, std=self.std_pix)]

            self.transform_augmentation_normalize = transforms.Compose(transforms_list_augmentation+transforms_list_normalize)
            self.data = Places205(root=env.PLACES205_DIR, split=self.split, transform=self.transform_augmentation_normalize)

        else:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label), index

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.epoch_size = len(dataset)
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.rand_seed = 0
        self.data_loader_ = self.get_iterator()

        self.inv_transform = transforms.Compose([
            Denormalize(self.dataset.mean_pix, self.dataset.std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

    def get_iterator(self):
        random.seed(self.rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the index of an image it returns the original image and its index in the dataset plus the label of the rotation, i.e., 0 for 0 degrees rotation, 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees. 4 rotated copies of the image will be created during neural network forward.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, _, index = self.dataset[idx]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                image_indices = torch.LongTensor([index, index, index, index])
                return img, rotation_labels, image_indices
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==3)
                batch_size, rotations = batch[1].size()
                batch[1] = batch[1].view([batch_size*rotations])
                batch[2] = batch[2].view([batch_size*rotations])
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the index of an image it returns the image and its categorical label and index in the dataset.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label, index = self.dataset[idx]
                return img, categorical_label, index
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list = range(self.epoch_size),
                                              load      = _load_function)
        data_loader = tnt_dataset.parallel(batch_size   = self.batch_size,
                                           collate_fn   = _collate_fun,
                                           num_workers  = self.num_workers,
                                           shuffle      = self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        self.rand_seed = epoch * self.epoch_size
        random.seed(self.rand_seed)
        return self.data_loader_

    def __len__(self):
        return len(self.data_loader_)
