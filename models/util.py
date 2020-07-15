import torch

import sys
sys.path.append('.')

import torch.nn as nn
import numpy as np

def gen_A(concur, sums, threshold=0.5, p=0.25, eps=1e-6):
    num_attribute = len(sums)
    sums = np.expand_dims(sums, axis=1)
    _adj = concur / sums
    _adj[_adj < threshold] = 0
    _adj[_adj >= threshold] = 1
    _adj = _adj * p / (_adj.sum(0, keepdims=True) + eps)
    _adj = _adj + (np.identity(num_attribute, np.int) * (1-p))
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

if __name__ == "__main__":
    from data import build_datasource
    datasource = build_datasource('ppe', root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    pass