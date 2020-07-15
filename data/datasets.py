import torch
import torch.nn as nn
import sys
sys.path.append('.')
import numpy as np
import torchvision.datasets as datasets

from PIL import Image

def imread(path):
    image = Image.open(path)
    return image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = imread(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)

class ImageDataset_GCN(torch.utils.data.Dataset):
    def __init__(self, data, inp, transform=None):
        self.data = data
        self.transform = transform
        self.inp = inp
        
    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = imread(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.inp), label
    
    def __len__(self):
        return len(self.data)