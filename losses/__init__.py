import sys
sys.path.append('.')
import torch
import torch.nn as nn

from .CE_loss import CEL_Sigmoid

def build_losses(config, pos_ratio, use_gpu=True):
    cfg_loss = config['loss']
    if cfg_loss['name'] == 'BCEWithLogitsLoss':
        pos_weight = torch.exp(-1 * pos_ratio)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight), {}
    elif cfg_loss['name'] == 'CEL_Sigmoid':
        return CEL_Sigmoid(pos_ratio, use_gpu=use_gpu), {}
    elif cfg_loss['name'] == 'MultiLabelSoftMarginLoss':
        return nn.MultiLabelSoftMarginLoss(), {}
    else:
        raise KeyError('config[loss] error')


if __name__ == "__main__":
    target = torch.ones([10, 64], dtype=torch.float32) # 64 classes, batch size = 10
    output = torch.full([10, 64], 1.5) # A prediction (logit)
    pos_weight = torch.ones([64]) # All weights are equal to 1
    criterion1 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_weights = pos_weight.expand(10, 64)
    criterion2 = torch.nn.BCEWithLogitsLoss(weight=bce_weights)
    out1 = criterion1(output, target)
    out2 = criterion2(output, target)
    pass