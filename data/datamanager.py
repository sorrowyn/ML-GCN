import os
import json
import torch
import torchvision
import sys
sys.path.append('.')

import torch.nn as nn
import numpy as np

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from .datasets import ImageDataset, ImageDataset_GCN
from .image import build_datasource
from .util import gen_M_N, word_embedding

class DataManger(object):
    def __init__(self, config, phase='train'):
        super().__init__()
        self.data_name = config['name']

        self.datasource = build_datasource(
            name=self.data_name,
            root_dir=config['data_dir'],
            download=config['download'],
            extract=config['extract'])

        self.inp = word_embedding(self.datasource.get_attribute(), dim=300)

        transform = dict()
        transform['train'] = transforms.Compose([
            transforms.Resize(size=self.datasource.get_image_size()),
            transforms.Pad(padding=10, fill=0, padding_mode='constant'),
            transforms.RandomCrop(size=self.datasource.get_image_size(), padding=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform['val'] = transforms.Compose([
            transforms.Resize(size=self.datasource.get_image_size()),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if 'test' in self.datasource.get_list_phase():
            transform['test'] = transforms.Compose([
                transforms.Resize(size=self.datasource.get_image_size()),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.dataset = dict()
        for _phase in self.datasource.get_list_phase():
            self.dataset[_phase] = ImageDataset_GCN(
                self.datasource.get_data(_phase),
                inp = self.inp,
                transform=transform[_phase])

        if phase == 'train':
            self.train_loader = DataLoader(
                dataset=self.dataset['train'],
                batch_size=config['batch_size'],
                shuffle=config['shuffle'],
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                drop_last=config['drop_last']
            )

            self.val_loader = DataLoader(
                dataset=self.dataset['val'],
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                drop_last=config['drop_last']
            )
        elif phase == 'test':
            if 'test' in self.datasource.get_list_phase():
                self.test_loader = DataLoader(self.dataset['test'], batch_size=32, shuffle=False, drop_last=False)
        else:
            raise ValueError("phase == train or phase == test")

    def get_dataloader(self, dataset):
        if dataset not in ['train', 'val', 'test']:
            raise ValueError("Error dataset paramaster, dataset in [train, val, test]")
        if dataset == 'train':
            return self.train_loader
        elif dataset == 'val':
            return self.val_loader
        elif dataset == 'test':
            return self.test_loader
    
    def get_M_N(self):
        return gen_M_N(self.datasource.get_data('train'), len(self.datasource.get_attribute()))